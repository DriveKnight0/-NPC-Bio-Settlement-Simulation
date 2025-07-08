import numpy as np
import cupy as cp
from numba import cuda, float32, int32
import math
import time
from typing import List, Tuple, Dict
import random

from rdr2_world_generator import RDR2WorldGenerator, RDR2WorldConfig, TERRAIN_COLORS
from rdr2_agent import RDR2Agent, AgentRole, AgentState, create_rdr2_agents

# CUDA kernel for trail evolution
@cuda.jit
def evolve_trail_kernel(trail_map, new_trail_map, height, width, evaporation_rate, diffusion_rate):
    """CUDA kernel for parallel trail evolution"""
    x, y = cuda.grid(2)
    
    if x >= width or y >= height:
        return
    
    # Get current trail value
    current_trail = trail_map[y, x]
    
    # Apply evaporation
    new_value = current_trail * (1.0 - evaporation_rate)
    
    # Apply diffusion
    total_neighbors = 0.0
    neighbor_sum = 0.0
    
    # Check all 8 neighbors
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            
            nx = x + dx
            ny = y + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                neighbor_sum += trail_map[ny, nx]
                total_neighbors += 1.0
    
    if total_neighbors > 0:
        # Add diffusion effect
        diffusion_amount = (neighbor_sum / total_neighbors - current_trail) * diffusion_rate
        new_value += diffusion_amount
    
    # Ensure non-negative
    new_trail_map[y, x] = max(0.0, new_value)

@cuda.jit
def agent_movement_kernel(agent_positions, agent_targets, agent_roles, agent_states, 
                         trail_map, terrain_map, width, height, num_agents, movement_speeds):
    """CUDA kernel for parallel agent movement"""
    idx = cuda.grid(1)
    
    if idx >= num_agents:
        return
    
    # Get agent data
    agent_x = agent_positions[idx, 0]
    agent_y = agent_positions[idx, 1]
    target_x = agent_targets[idx, 0]
    target_y = agent_targets[idx, 1]
    role = agent_roles[idx]
    
    # Skip if no target
    if target_x < 0 or target_y < 0:
        return
    
    # Calculate movement direction
    dx = target_x - agent_x
    dy = target_y - agent_y
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance < 1.0:
        # Reached target
        agent_positions[idx, 0] = target_x
        agent_positions[idx, 1] = target_y
        agent_targets[idx, 0] = -1
        agent_targets[idx, 1] = -1
        return
    
    # Normalize direction
    dx /= distance
    dy /= distance
    
    # Apply movement speed
    speed = movement_speeds[role]
    dx *= speed
    dy *= speed
    
    # Calculate new position
    new_x = agent_x + dx
    new_y = agent_y + dy
    
    # Clamp to bounds
    new_x = max(0.0, min(float(width - 1), new_x))
    new_y = max(0.0, min(float(height - 1), new_y))
    
    # Update position
    agent_positions[idx, 0] = new_x
    agent_positions[idx, 1] = new_y

@cuda.jit
def trail_deposit_kernel(agent_positions, agent_roles, trail_map, width, height, 
                        num_agents, trail_strengths):
    """CUDA kernel for parallel trail deposition"""
    idx = cuda.grid(1)
    
    if idx >= num_agents:
        return
    
    # Get agent position and role
    x = int(agent_positions[idx, 0])
    y = int(agent_positions[idx, 1])
    role = agent_roles[idx]
    
    # Check bounds
    if 0 <= x < width and 0 <= y < height:
        # Deposit trail based on role
        strength = trail_strengths[role]
        
        # Atomic add to avoid race conditions
        cuda.atomic.add(trail_map, (y, x), strength)

class RDR2GPUSimulation:
    def __init__(self, config: RDR2WorldConfig = None, num_agents: int = 1000):
        self.config = config or RDR2WorldConfig()
        self.num_agents = num_agents
        
        # Initialize world
        print("Generating RDR2-inspired world...")
        self.world_generator = RDR2WorldGenerator(self.config)
        self.terrain = self.world_generator.generate_world()
        
        # Initialize agents
        print(f"Creating {num_agents} agents...")
        self.agents = create_rdr2_agents(self.world_generator, num_agents)
        
        # GPU setup
        self._setup_gpu_arrays()
        
        # Simulation parameters
        self.evaporation_rate = 0.02
        self.diffusion_rate = 0.1
        self.trail_deposit_rate = 1.0
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0.0
        
        print(f"RDR2 GPU Simulation initialized:")
        print(f"  World size: {self.config.width}x{self.config.height}")
        print(f"  Agents: {len(self.agents)}")
        print(f"  Cities: {len(self.world_generator.cities)}")
        print(f"  GPU arrays allocated")
    
    def _setup_gpu_arrays(self):
        """Setup GPU arrays for simulation"""
        # Trail map
        self.trail_map_gpu = cp.zeros((self.config.height, self.config.width), dtype=cp.float32)
        self.new_trail_map_gpu = cp.zeros((self.config.height, self.config.width), dtype=cp.float32)
        
        # Terrain map
        self.terrain_map_gpu = cp.array(self.terrain, dtype=cp.int32)
        
        # Agent data arrays
        self.agent_positions_gpu = cp.zeros((len(self.agents), 2), dtype=cp.float32)
        self.agent_targets_gpu = cp.zeros((len(self.agents), 2), dtype=cp.float32)
        self.agent_roles_gpu = cp.zeros(len(self.agents), dtype=cp.int32)
        self.agent_states_gpu = cp.zeros(len(self.agents), dtype=cp.int32)
        
        # Initialize agent data
        for i, agent in enumerate(self.agents):
            self.agent_positions_gpu[i, 0] = agent.x
            self.agent_positions_gpu[i, 1] = agent.y
            self.agent_targets_gpu[i, 0] = agent.target_x or -1
            self.agent_targets_gpu[i, 1] = agent.target_y or -1
            self.agent_roles_gpu[i] = list(AgentRole).index(agent.role)
            self.agent_states_gpu[i] = list(AgentState).index(agent.state)
        
        # Movement speed lookup table
        speed_table = {
            AgentRole.EXPLORER: 1.2,
            AgentRole.TRADER: 0.8,
            AgentRole.SETTLER: 0.6,
            AgentRole.MINER: 0.7,
            AgentRole.FARMER: 0.6,
            AgentRole.HUNTER: 1.1,
            AgentRole.OUTLAW: 1.3,
            AgentRole.LAWMAN: 1.0
        }
        speeds = [speed_table[role] for role in AgentRole]
        self.movement_speeds_gpu = cp.array(speeds, dtype=cp.float32)
        
        # Trail strength lookup table
        strength_table = {
            AgentRole.EXPLORER: 1.2,
            AgentRole.TRADER: 1.5,
            AgentRole.SETTLER: 0.8,
            AgentRole.MINER: 0.6,
            AgentRole.FARMER: 0.7,
            AgentRole.HUNTER: 0.9,
            AgentRole.OUTLAW: 0.5,
            AgentRole.LAWMAN: 1.1
        }
        strengths = [strength_table[role] for role in AgentRole]
        self.trail_strengths_gpu = cp.array(strengths, dtype=cp.float32)
        
        # CUDA kernel configuration
        self.threads_per_block_2d = (16, 16)
        self.blocks_per_grid_2d = (
            (self.config.width + self.threads_per_block_2d[0] - 1) // self.threads_per_block_2d[0],
            (self.config.height + self.threads_per_block_2d[1] - 1) // self.threads_per_block_2d[1]
        )
        
        self.threads_per_block_1d = 256
        self.blocks_per_grid_1d = (len(self.agents) + self.threads_per_block_1d - 1) // self.threads_per_block_1d
    
    def update(self):
        """Update simulation for one step"""
        start_time = time.time()
        
        # Update agent behavior on CPU
        self._update_agent_behavior()
        
        # Move agents on GPU
        self._update_agent_movement()
        
        # Deposit trails on GPU
        self._deposit_trails()
        
        # Evolve trail map on GPU
        self._evolve_trails()
        
        # Update frame statistics
        self.frame_count += 1
        frame_time = time.time() - start_time
        self.total_time += frame_time
        
        if self.frame_count % 100 == 0:
            avg_fps = self.frame_count / self.total_time
            print(f"Frame {self.frame_count}: {avg_fps:.1f} FPS (avg), {1000*frame_time:.1f}ms")
    
    def _update_agent_behavior(self):
        """Update agent behavior on CPU"""
        # Download agent positions from GPU
        positions = cp.asnumpy(self.agent_positions_gpu)
        
        # Update agent logic
        for i, agent in enumerate(self.agents):
            agent.x = int(positions[i, 0])
            agent.y = int(positions[i, 1])
            
            # Update agent behavior (simplified for GPU performance)
            self._update_single_agent_behavior(agent)
            
            # Upload new targets
            if agent.target_x is not None and agent.target_y is not None:
                self.agent_targets_gpu[i, 0] = agent.target_x
                self.agent_targets_gpu[i, 1] = agent.target_y
            else:
                self.agent_targets_gpu[i, 0] = -1
                self.agent_targets_gpu[i, 1] = -1
    
    def _update_single_agent_behavior(self, agent: RDR2Agent):
        """Simplified agent behavior update for GPU performance"""
        # Simplified behavior logic for performance
        if agent.target_x is None or agent.target_y is None or self._reached_target(agent):
            # Set new random target based on role
            if agent.role == AgentRole.EXPLORER:
                # Seek unexplored areas
                agent.target_x = random.randint(10, self.config.width - 10)
                agent.target_y = random.randint(10, self.config.height - 10)
            
            elif agent.role == AgentRole.TRADER:
                # Head to nearest city
                if self.world_generator.cities:
                    city = random.choice(self.world_generator.cities)
                    agent.target_x = city.x
                    agent.target_y = city.y
            
            elif agent.role in [AgentRole.SETTLER, AgentRole.FARMER]:
                # Seek plains or river areas
                attempts = 0
                while attempts < 10:
                    test_x = random.randint(20, self.config.width - 20)
                    test_y = random.randint(20, self.config.height - 20)
                    
                    from rdr2_world_generator import PLAINS, RIVER
                    if self.terrain[test_y, test_x] in [PLAINS, RIVER]:
                        agent.target_x = test_x
                        agent.target_y = test_y
                        break
                    attempts += 1
                
                if attempts >= 10:
                    agent.target_x = random.randint(10, self.config.width - 10)
                    agent.target_y = random.randint(10, self.config.height - 10)
            
            elif agent.role == AgentRole.MINER:
                # Seek mountain areas
                attempts = 0
                while attempts < 10:
                    test_x = random.randint(10, self.config.width - 10)
                    test_y = random.randint(10, self.config.height - 10)
                    
                    from rdr2_world_generator import MOUNTAIN, FOOTHILLS
                    if self.terrain[test_y, test_x] in [MOUNTAIN, FOOTHILLS]:
                        agent.target_x = test_x
                        agent.target_y = test_y
                        break
                    attempts += 1
                
                if attempts >= 10:
                    agent.target_x = random.randint(10, self.config.width - 10)
                    agent.target_y = random.randint(10, self.config.height - 10)
            
            elif agent.role == AgentRole.HUNTER:
                # Follow areas with moderate trail activity
                trail_cpu = cp.asnumpy(self.trail_map_gpu)
                best_trail = 0
                best_target = None
                
                for _ in range(5):
                    test_x = agent.x + random.randint(-20, 20)
                    test_y = agent.y + random.randint(-20, 20)
                    
                    if (0 <= test_x < self.config.width and 
                        0 <= test_y < self.config.height):
                        trail_value = trail_cpu[test_y, test_x]
                        
                        if 5 < trail_value < 20 and trail_value > best_trail:
                            best_trail = trail_value
                            best_target = (test_x, test_y)
                
                if best_target:
                    agent.target_x, agent.target_y = best_target
                else:
                    agent.target_x = random.randint(10, self.config.width - 10)
                    agent.target_y = random.randint(10, self.config.height - 10)
            
            else:
                # Default random movement
                agent.target_x = random.randint(10, self.config.width - 10)
                agent.target_y = random.randint(10, self.config.height - 10)
    
    def _reached_target(self, agent: RDR2Agent) -> bool:
        """Check if agent reached target"""
        if agent.target_x is None or agent.target_y is None:
            return True
        
        distance = math.sqrt((agent.x - agent.target_x)**2 + (agent.y - agent.target_y)**2)
        return distance < 2.0
    
    def _update_agent_movement(self):
        """Update agent movement on GPU"""
        agent_movement_kernel[self.blocks_per_grid_1d, self.threads_per_block_1d](
            self.agent_positions_gpu,
            self.agent_targets_gpu,
            self.agent_roles_gpu,
            self.agent_states_gpu,
            self.trail_map_gpu,
            self.terrain_map_gpu,
            self.config.width,
            self.config.height,
            len(self.agents),
            self.movement_speeds_gpu
        )
        
        cuda.synchronize()
    
    def _deposit_trails(self):
        """Deposit agent trails on GPU"""
        trail_deposit_kernel[self.blocks_per_grid_1d, self.threads_per_block_1d](
            self.agent_positions_gpu,
            self.agent_roles_gpu,
            self.trail_map_gpu,
            self.config.width,
            self.config.height,
            len(self.agents),
            self.trail_strengths_gpu
        )
        
        cuda.synchronize()
    
    def _evolve_trails(self):
        """Evolve trail map on GPU"""
        evolve_trail_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
            self.trail_map_gpu,
            self.new_trail_map_gpu,
            self.config.height,
            self.config.width,
            self.evaporation_rate,
            self.diffusion_rate
        )
        
        cuda.synchronize()
        
        # Swap trail maps
        self.trail_map_gpu, self.new_trail_map_gpu = self.new_trail_map_gpu, self.trail_map_gpu
    
    def get_visualization_data(self) -> Tuple[np.ndarray, np.ndarray, List[RDR2Agent]]:
        """Get data for visualization"""
        # Download data from GPU
        terrain_cpu = cp.asnumpy(self.terrain_map_gpu)
        trail_cpu = cp.asnumpy(self.trail_map_gpu)
        positions_cpu = cp.asnumpy(self.agent_positions_gpu)
        
        # Update agent positions
        for i, agent in enumerate(self.agents):
            agent.x = int(positions_cpu[i, 0])
            agent.y = int(positions_cpu[i, 1])
        
        return terrain_cpu, trail_cpu, self.agents
    
    def get_world_generator(self) -> RDR2WorldGenerator:
        """Get the world generator"""
        return self.world_generator
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        positions = cp.asnumpy(self.agent_positions_gpu)
        trail_sum = float(cp.sum(self.trail_map_gpu))
        trail_max = float(cp.max(self.trail_map_gpu))
        
        # Calculate agent distribution by role
        role_counts = {}
        for agent in self.agents:
            role_counts[agent.role.value] = role_counts.get(agent.role.value, 0) + 1
        
        # Calculate coverage
        active_cells = int(cp.sum(self.trail_map_gpu > 0.1))
        total_cells = self.config.width * self.config.height
        coverage = active_cells / total_cells
        
        return {
            'frame': self.frame_count,
            'fps': self.frame_count / self.total_time if self.total_time > 0 else 0,
            'agents': len(self.agents),
            'trail_sum': trail_sum,
            'trail_max': trail_max,
            'coverage': coverage,
            'role_distribution': role_counts,
            'cities': len(self.world_generator.cities),
            'world_size': f"{self.config.width}x{self.config.height}"
        }

def create_rdr2_gpu_simulation(num_agents: int = 1000) -> RDR2GPUSimulation:
    """Create RDR2 GPU simulation with specified number of agents"""
    config = RDR2WorldConfig(
        width=250,
        height=250,
        num_cities=7,
        num_towns=15,
        num_settlements=25,
        num_mountain_ranges=3,
        num_major_rivers=4,
        num_minor_rivers=8
    )
    
    return RDR2GPUSimulation(config, num_agents)

if __name__ == "__main__":
    # Test the simulation
    print("Creating RDR2 GPU Simulation...")
    sim = create_rdr2_gpu_simulation(1000)
    
    print("\nRunning simulation test...")
    for i in range(100):
        sim.update()
        
        if i % 20 == 0:
            stats = sim.get_statistics()
            print(f"Step {i}: {stats['fps']:.1f} FPS, Trail sum: {stats['trail_sum']:.1f}, Coverage: {stats['coverage']:.3f}")
    
    print("\nSimulation test completed!") 
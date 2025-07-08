import numpy as np
import random
import math
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque

class AgentRole(Enum):
    EXPLORER = "explorer"
    TRADER = "trader"
    SETTLER = "settler"
    MINER = "miner"
    FARMER = "farmer"
    HUNTER = "hunter"
    OUTLAW = "outlaw"
    LAWMAN = "lawman"

class AgentState(Enum):
    EXPLORING = "exploring"
    TRAVELING = "traveling"
    SETTLING = "settling"
    WORKING = "working"
    TRADING = "trading"
    RESTING = "resting"
    FLEEING = "fleeing"

@dataclass
class AgentStats:
    health: float = 100.0
    energy: float = 100.0
    hunger: float = 0.0
    wealth: float = 100.0
    reputation: float = 0.0
    experience: float = 0.0

@dataclass
class AgentMemory:
    visited_locations: set
    known_settlements: list
    known_resources: dict
    danger_zones: set
    trade_routes: list
    last_settlement: tuple

class PhysarumTrailSystem:
    """Enhanced trail system using physarum-inspired algorithms"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.trail_map = np.zeros((height, width), dtype=np.float32)
        self.efficiency_map = np.zeros((height, width), dtype=np.float32)
        self.city_attractions = np.zeros((height, width), dtype=np.float32)
        
        # Physarum parameters
        self.deposit_rate = 1.0
        self.decay_rate = 0.995
        self.diffusion_rate = 0.05
        self.reinforcement_factor = 1.5
    
    def deposit_trail(self, x: int, y: int, strength: float, agent_role: AgentRole):
        """Deposit trail based on agent role"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Different roles deposit different strength trails
            role_multipliers = {
                AgentRole.TRADER: 2.0,      # Traders create strong trade routes
                AgentRole.EXPLORER: 0.8,    # Explorers create weak trails
                AgentRole.SETTLER: 1.2,     # Settlers create moderate trails
                AgentRole.LAWMAN: 1.5,      # Lawmen create patrol routes
                AgentRole.OUTLAW: 0.3,      # Outlaws avoid leaving trails
            }
            
            multiplier = role_multipliers.get(agent_role, 1.0)
            self.trail_map[y, x] += strength * multiplier
    
    def get_trail_strength(self, x: int, y: int) -> float:
        """Get trail strength at position"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.trail_map[y, x]
        return 0.0
    
    def update_trails(self):
        """Update trail system using physarum dynamics"""
        # Apply decay
        self.trail_map *= self.decay_rate
        
        # Apply diffusion (simple convolution)
        if np.max(self.trail_map) > 0:
            # Create diffusion kernel
            kernel = np.array([[0.05, 0.1, 0.05],
                              [0.1,  0.6, 0.1 ],
                              [0.05, 0.1, 0.05]])
            
            # Apply convolution for diffusion
            from scipy.ndimage import convolve
            diffused = convolve(self.trail_map, kernel, mode='constant', cval=0.0)
            self.trail_map = (1 - self.diffusion_rate) * self.trail_map + self.diffusion_rate * diffused
    
    def reinforce_successful_path(self, path: List[Tuple[int, int]], agent_role: AgentRole):
        """Reinforce a successful path between cities"""
        for x, y in path:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.trail_map[y, x] *= self.reinforcement_factor
                self.efficiency_map[y, x] += 0.1
    
    def set_city_attraction(self, cities):
        """Set attraction fields around cities"""
        self.city_attractions.fill(0.0)
        
        for city in cities:
            cx, cy = int(city.x), int(city.y)
            
            # Create attraction field around city
            for dy in range(-15, 16):
                for dx in range(-15, 16):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance > 0:
                            attraction = city.population / (100.0 * (distance + 1))
                            self.city_attractions[y, x] += attraction

class RDR2Agent:
    def __init__(self, x: int, y: int, agent_id: int, role: AgentRole = None):
        self.id = agent_id
        self.x = x
        self.y = y
        self.trail_x = float(x)
        self.trail_y = float(y)
        
        # Agent characteristics
        self.role = role or random.choice(list(AgentRole))
        self.state = AgentState.EXPLORING
        self.stats = AgentStats()
        self.memory = AgentMemory(
            visited_locations=set(),
            known_settlements=[],
            known_resources={},
            danger_zones=set(),
            trade_routes=[],
            last_settlement=None
        )
        
        # Movement and behavior
        self.target_x = None
        self.target_y = None
        self.path = [(x, y)]
        self.stuck_counter = 0
        self.rest_timer = 0
        self.work_timer = 0
        self.trade_timer = 0
        
        # Physarum-inspired attributes
        self.trail_memory = deque(maxlen=50)  # Remember recent trail strengths
        self.successful_paths = []  # Store successful routes
        self.learning_rate = 0.05
        self.exploration_bonus = 0.3
        self.trail_following_strength = self._get_trail_following_strength()
        self.pathfinding_efficiency = 0.5
        
        # Role-specific attributes
        self.inventory = {}
        self.group_id = None
        self.preferred_terrain = self._get_preferred_terrain()
        self.detection_range = self._get_detection_range()
        self.movement_speed = self._get_movement_speed()
        self.aggression = self._get_aggression_level()
        
        # Trail deposits
        self.trail_strength = 1.0
        self.last_trail_deposit = 0
        
        # Initialize role-specific stats
        self._initialize_role_stats()
    
    def _get_trail_following_strength(self) -> float:
        """Get how much this agent follows existing trails"""
        strengths = {
            AgentRole.TRADER: 0.9,      # Heavily follows established trade routes
            AgentRole.LAWMAN: 0.7,      # Follows patrol routes and roads
            AgentRole.SETTLER: 0.6,     # Moderately follows trails to safe areas
            AgentRole.FARMER: 0.6,      # Follows routes to markets
            AgentRole.HUNTER: 0.4,      # Sometimes follows game trails
            AgentRole.MINER: 0.5,       # Follows routes to mining areas
            AgentRole.EXPLORER: 0.2,    # Prefers unexplored areas
            AgentRole.OUTLAW: 0.1,      # Avoids well-traveled routes
        }
        return strengths.get(self.role, 0.5)
    
    def calculate_physarum_attractiveness(self, pos: Tuple[int, int], 
                                         trail_system: PhysarumTrailSystem, 
                                         cities) -> float:
        """Calculate position attractiveness using physarum-inspired algorithm"""
        x, y = pos
        
        # Base trail strength
        trail_strength = trail_system.get_trail_strength(x, y)
        
        # City attraction
        city_attraction = trail_system.city_attractions[y, x] if y < trail_system.height and x < trail_system.width else 0
        
        # Distance to target (if any)
        distance_score = 0.0
        if self.target_x is not None and self.target_y is not None:
            distance = math.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
            max_distance = math.sqrt(trail_system.width**2 + trail_system.height**2)
            distance_score = 1.0 - (distance / max_distance)
        
        # Exploration bonus (prefer unvisited areas for explorers)
        exploration_score = 0.0
        if self.role == AgentRole.EXPLORER:
            if (x, y) not in self.memory.visited_locations:
                exploration_score = self.exploration_bonus
        
        # Anti-oscillation (avoid recent positions)
        recent_penalty = 0.0
        if len(self.path) > 3:
            for recent_pos in self.path[-3:]:
                if recent_pos == (x, y):
                    recent_penalty = -0.5
                    break
        
        # Role-specific scoring
        role_bonus = 0.0
        if self.role == AgentRole.TRADER:
            # Traders prefer routes between cities
            role_bonus = city_attraction * 0.5
        elif self.role == AgentRole.OUTLAW:
            # Outlaws avoid high-traffic areas
            role_bonus = -trail_strength * 0.3
        elif self.role == AgentRole.LAWMAN:
            # Lawmen prefer patrolling established routes
            role_bonus = trail_strength * 0.2
        
        # Combine all factors
        total_attractiveness = (
            self.trail_following_strength * trail_strength +
            0.4 * distance_score +
            0.3 * city_attraction +
            exploration_score +
            role_bonus +
            recent_penalty
        )
        
        return max(0.0, total_attractiveness)
    
    def physarum_pathfinding(self, world_generator, trail_system: PhysarumTrailSystem) -> Tuple[int, int]:
        """Use physarum-inspired pathfinding to choose next move"""
        
        # Get valid neighbors
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                new_x, new_y = self.x + dx, self.y + dy
                
                # Check bounds
                if (0 <= new_x < world_generator.config.width and 
                    0 <= new_y < world_generator.config.height):
                    
                    # Check terrain (avoid water/mountains for most agents)
                    terrain = world_generator.terrain[new_y, new_x]
                    from rdr2_world_generator import WATER, MOUNTAIN
                    
                    # Allow different movement for different roles
                    if terrain == WATER and self.role not in [AgentRole.EXPLORER]:
                        continue
                    if terrain == MOUNTAIN and self.role not in [AgentRole.EXPLORER, AgentRole.MINER]:
                        continue
                    
                    neighbors.append((new_x, new_y))
        
        if not neighbors:
            return self.x, self.y
        
        # Calculate attractiveness for each neighbor
        attractiveness_scores = []
        for pos in neighbors:
            score = self.calculate_physarum_attractiveness(pos, trail_system, world_generator.cities)
            attractiveness_scores.append(score)
        
        # Choose move based on attractiveness (with some randomness)
        if len(attractiveness_scores) > 0 and max(attractiveness_scores) > 0:
            try:
                # Probabilistic selection based on scores
                total_score = sum(attractiveness_scores)
                if total_score > 0:
                    probabilities = [score / total_score for score in attractiveness_scores]
                    
                    # Add some randomness for exploration
                    if self.role == AgentRole.EXPLORER:
                        # Explorers are more random
                        for i in range(len(probabilities)):
                            probabilities[i] = 0.7 * probabilities[i] + 0.3 / len(probabilities)
                    
                    # Ensure probabilities sum to 1.0 (fix numerical precision issues)
                    prob_sum = sum(probabilities)
                    if prob_sum > 0:
                        probabilities = [p / prob_sum for p in probabilities]
                        
                        # Final check for valid probabilities
                        if abs(sum(probabilities) - 1.0) < 1e-10 and all(p >= 0 for p in probabilities):
                            choice_index = np.random.choice(len(neighbors), p=probabilities)
                            return neighbors[choice_index]
            except (ValueError, ZeroDivisionError):
                # Fall through to random selection
                pass
        
        # Fallback to random movement
        return random.choice(neighbors) if neighbors else (self.x, self.y)
    
    def update(self, world_generator, agents: List['RDR2Agent'], trail_system: PhysarumTrailSystem):
        """Enhanced update with physarum-based movement"""
        
        # Update basic stats
        self._update_stats()
        
        # Update memory
        self._update_memory(world_generator)
        
        # Set target based on role and current state
        self._set_intelligent_target(world_generator, trail_system)
        
        # Use physarum pathfinding for movement
        old_x, old_y = self.x, self.y
        new_x, new_y = self.physarum_pathfinding(world_generator, trail_system)
        
        # Move if position changed
        if (new_x, new_y) != (old_x, old_y):
            self.x, self.y = new_x, new_y
            self.path.append((self.x, self.y))
            
            # Deposit trail
            trail_system.deposit_trail(self.x, self.y, self.trail_strength, self.role)
            
            # Update trail memory
            trail_strength = trail_system.get_trail_strength(self.x, self.y)
            self.trail_memory.append(trail_strength)
            
            # Learn from trail efficiency
            if len(self.trail_memory) > 10:
                recent_avg = np.mean(list(self.trail_memory)[-10:])
                if recent_avg > 5.0:  # Following successful trails
                    self.pathfinding_efficiency = min(1.0, self.pathfinding_efficiency + self.learning_rate)
        
        # Handle interactions
        self._handle_interactions(agents, world_generator)
    
    def _set_intelligent_target(self, world_generator, trail_system: PhysarumTrailSystem):
        """Set intelligent targets based on role and trail information"""
        
        # Check if current target is still valid
        if (self.target_x is not None and self.target_y is not None and
            abs(self.x - self.target_x) < 3 and abs(self.y - self.target_y) < 3):
            # Reached target
            self.target_x = self.target_y = None
            
            # Reinforce successful path if it was to a city
            if len(self.path) > 5:
                for city in world_generator.cities:
                    if abs(city.x - self.x) < 5 and abs(city.y - self.y) < 5:
                        trail_system.reinforce_successful_path(self.path[-20:], self.role)
                        break
        
        # Set new target if needed
        if self.target_x is None or self.target_y is None:
            if self.role == AgentRole.TRADER:
                # Traders move between cities
                if len(world_generator.cities) > 1:
                    # Choose a different city from current location
                    available_cities = [city for city in world_generator.cities 
                                      if abs(city.x - self.x) > 20 or abs(city.y - self.y) > 20]
                    if available_cities:
                        target_city = random.choice(available_cities)
                        self.target_x, self.target_y = target_city.x, target_city.y
            
            elif self.role == AgentRole.EXPLORER:
                # Explorers seek unexplored areas
                attempts = 0
                while attempts < 10:
                    test_x = random.randint(20, world_generator.config.width - 20)
                    test_y = random.randint(20, world_generator.config.height - 20)
                    
                    if ((test_x, test_y) not in self.memory.visited_locations and
                        trail_system.get_trail_strength(test_x, test_y) < 1.0):
                        self.target_x, self.target_y = test_x, test_y
                        break
                    attempts += 1
            
            elif self.role in [AgentRole.SETTLER, AgentRole.FARMER]:
                # Seek good terrain near water sources
                attempts = 0
                while attempts < 10:
                    test_x = random.randint(20, world_generator.config.width - 20)
                    test_y = random.randint(20, world_generator.config.height - 20)
                    
                    from rdr2_world_generator import PLAINS, RIVER
                    if world_generator.terrain[test_y, test_x] in [PLAINS, RIVER]:
                        self.target_x, self.target_y = test_x, test_y
                        break
                    attempts += 1
            
            # Fallback random target
            if self.target_x is None:
                self.target_x = random.randint(10, world_generator.config.width - 10)
                self.target_y = random.randint(10, world_generator.config.height - 10)
    
    def _get_preferred_terrain(self) -> List[int]:
        """Get preferred terrain types based on role"""
        from rdr2_world_generator import PLAINS, FOREST, MOUNTAIN, DESERT, SWAMP, RIVER, CITY, TOWN, SETTLEMENT, ROAD, RAILWAY, FOOTHILLS
        
        preferences = {
            AgentRole.EXPLORER: [PLAINS, FOREST, MOUNTAIN],
            AgentRole.TRADER: [CITY, TOWN, SETTLEMENT, ROAD, RAILWAY],
            AgentRole.SETTLER: [PLAINS, FOREST, RIVER],
            AgentRole.MINER: [MOUNTAIN, FOOTHILLS],
            AgentRole.FARMER: [PLAINS, RIVER],
            AgentRole.HUNTER: [FOREST, PLAINS],
            AgentRole.OUTLAW: [DESERT, MOUNTAIN, FOREST],
            AgentRole.LAWMAN: [CITY, TOWN, SETTLEMENT, ROAD]
        }
        return preferences.get(self.role, [PLAINS, FOREST])
    
    def _get_detection_range(self) -> int:
        """Get detection range based on role"""
        ranges = {
            AgentRole.EXPLORER: 8,
            AgentRole.TRADER: 6,
            AgentRole.SETTLER: 5,
            AgentRole.MINER: 4,
            AgentRole.FARMER: 5,
            AgentRole.HUNTER: 10,
            AgentRole.OUTLAW: 12,
            AgentRole.LAWMAN: 15
        }
        return ranges.get(self.role, 6)
    
    def _get_movement_speed(self) -> float:
        """Get movement speed based on role"""
        speeds = {
            AgentRole.EXPLORER: 1.5,
            AgentRole.TRADER: 1.0,
            AgentRole.SETTLER: 0.8,
            AgentRole.MINER: 0.9,
            AgentRole.FARMER: 0.8,
            AgentRole.HUNTER: 1.2,
            AgentRole.OUTLAW: 1.8,
            AgentRole.LAWMAN: 1.4
        }
        return speeds.get(self.role, 1.0)
    
    def _get_aggression_level(self) -> float:
        """Get aggression level based on role"""
        aggression = {
            AgentRole.EXPLORER: 0.2,
            AgentRole.TRADER: 0.1,
            AgentRole.SETTLER: 0.1,
            AgentRole.MINER: 0.3,
            AgentRole.FARMER: 0.1,
            AgentRole.HUNTER: 0.4,
            AgentRole.OUTLAW: 0.9,
            AgentRole.LAWMAN: 0.7
        }
        return aggression.get(self.role, 0.2)
    
    def _initialize_role_stats(self):
        """Initialize stats based on role"""
        if self.role == AgentRole.EXPLORER:
            self.stats.health = 90.0
            self.stats.energy = 100.0
            self.stats.wealth = 50.0
        elif self.role == AgentRole.TRADER:
            self.stats.wealth = 200.0
            self.stats.reputation = 10.0
            self.inventory['goods'] = 5
        elif self.role == AgentRole.SETTLER:
            self.stats.health = 80.0
            self.stats.energy = 90.0
            self.inventory['tools'] = 3
        elif self.role == AgentRole.MINER:
            self.stats.health = 100.0
            self.stats.energy = 120.0
            self.inventory['pickaxe'] = 1
        elif self.role == AgentRole.FARMER:
            self.stats.health = 85.0
            self.inventory['seeds'] = 5
        elif self.role == AgentRole.HUNTER:
            self.stats.health = 95.0
            self.stats.energy = 95.0
            self.inventory['rifle'] = 1
        elif self.role == AgentRole.OUTLAW:
            self.stats.health = 90.0
            self.stats.energy = 110.0
            self.stats.reputation = -20.0
            self.inventory['weapon'] = 1
        elif self.role == AgentRole.LAWMAN:
            self.stats.health = 100.0
            self.stats.reputation = 15.0
            self.inventory['badge'] = 1
    
    def _update_stats(self):
        """Update agent statistics"""
        # Energy decay
        self.stats.energy = max(0, self.stats.energy - 0.1)
        
        # Hunger increase
        self.stats.hunger = min(100, self.stats.hunger + 0.05)
        
        # Health effects from hunger and energy
        if self.stats.hunger > 80 or self.stats.energy < 20:
            self.stats.health = max(0, self.stats.health - 0.1)
        elif self.stats.hunger < 20 and self.stats.energy > 80:
            self.stats.health = min(100, self.stats.health + 0.05)
        
        # Experience gain
        self.stats.experience += 0.01
    
    def _update_memory(self, world_generator):
        """Update agent's memory with current observations"""
        # Remember current location
        self.memory.visited_locations.add((self.x, self.y))
        
        # Detect nearby cities
        for city in world_generator.cities:
            distance = math.sqrt((self.x - city.x)**2 + (self.y - city.y)**2)
            if distance < 10:
                city_info = (city.x, city.y, city.name)
                if city_info not in self.memory.known_settlements:
                    self.memory.known_settlements.append(city_info)
                    self.memory.last_settlement = (city.x, city.y)
        
        # Detect terrain resources
        if hasattr(world_generator, 'terrain'):
            terrain = world_generator.terrain[self.y, self.x]
            from rdr2_world_generator import MOUNTAIN, RIVER, FOREST
            
            if terrain == MOUNTAIN and 'mountain' not in self.memory.known_resources:
                self.memory.known_resources['mountain'] = []
            if terrain == RIVER and 'water' not in self.memory.known_resources:
                self.memory.known_resources['water'] = []
            if terrain == FOREST and 'forest' not in self.memory.known_resources:
                self.memory.known_resources['forest'] = []
            
            # Add current location to relevant resource memory
            if terrain in [MOUNTAIN, RIVER, FOREST]:
                resource_type = {MOUNTAIN: 'mountain', RIVER: 'water', FOREST: 'forest'}[terrain]
                if (self.x, self.y) not in self.memory.known_resources.get(resource_type, []):
                    if resource_type not in self.memory.known_resources:
                        self.memory.known_resources[resource_type] = []
                    self.memory.known_resources[resource_type].append((self.x, self.y))
    
    def _handle_interactions(self, agents: List['RDR2Agent'], world_generator):
        """Handle interactions with other agents"""
        nearby_agents = []
        
        # Find nearby agents
        for other_agent in agents:
            if other_agent.id != self.id:
                distance = math.sqrt((self.x - other_agent.x)**2 + (self.y - other_agent.y)**2)
                if distance < 5:
                    nearby_agents.append(other_agent)
        
        # Handle role-specific interactions
        for other_agent in nearby_agents:
            self._handle_agent_interaction(other_agent)

    def _handle_agent_interaction(self, other_agent: 'RDR2Agent'):
        """Handle interaction with a specific agent"""
        distance = math.sqrt((self.x - other_agent.x)**2 + (self.y - other_agent.y)**2)
        
        # Outlaw vs Lawman
        if self.role == AgentRole.OUTLAW and other_agent.role == AgentRole.LAWMAN:
            # Outlaw flees
            if distance < 3:
                self.stats.reputation -= 2.0
                other_agent.stats.reputation += 1.0
        
        # Trader interactions
        elif self.role == AgentRole.TRADER and other_agent.role in [AgentRole.SETTLER, AgentRole.FARMER]:
            # Trade opportunity
            if distance < 2 and random.random() < 0.1:
                self.stats.wealth += random.randint(5, 15)
                other_agent.stats.wealth -= random.randint(1, 5)
                self.stats.reputation += 0.5
        
        # Lawman helping citizens
        elif self.role == AgentRole.LAWMAN and other_agent.role in [AgentRole.SETTLER, AgentRole.FARMER, AgentRole.TRADER]:
            if distance < 2:
                other_agent.stats.reputation += 0.2
                self.stats.reputation += 0.1
        
        # Hunter vs Outlaw
        elif self.role == AgentRole.HUNTER and other_agent.role == AgentRole.OUTLAW:
            if distance < 2:
                other_agent.stats.reputation -= 1.0
    
    def get_role_color(self) -> Tuple[int, int, int]:
        """Get color for this agent based on role"""
        colors = {
            AgentRole.EXPLORER: (255, 180, 50),    # Orange
            AgentRole.TRADER: (50, 255, 50),       # Green
            AgentRole.SETTLER: (100, 150, 255),    # Blue
            AgentRole.MINER: (150, 75, 0),         # Brown
            AgentRole.FARMER: (255, 255, 100),     # Yellow
            AgentRole.HUNTER: (0, 150, 0),         # Dark Green
            AgentRole.OUTLAW: (255, 50, 50),       # Red
            AgentRole.LAWMAN: (100, 100, 255),     # Light Blue
        }
        return colors.get(self.role, (255, 255, 255))

def create_rdr2_agents(world_generator, num_agents: int = 1000) -> List[RDR2Agent]:
    """Create a population of RDR2 agents"""
    agents = []
    
    # Role distribution (inspired by RDR2 population)
    role_weights = {
        AgentRole.EXPLORER: 0.15,
        AgentRole.TRADER: 0.10,
        AgentRole.SETTLER: 0.25,
        AgentRole.MINER: 0.12,
        AgentRole.FARMER: 0.20,
        AgentRole.HUNTER: 0.08,
        AgentRole.OUTLAW: 0.05,
        AgentRole.LAWMAN: 0.05
    }
    
    # Create agents
    for i in range(num_agents):
        # Choose role based on weights
        role = np.random.choice(list(role_weights.keys()), p=list(role_weights.values()))
        
        # Find suitable starting position
        attempts = 0
        while attempts < 50:
            x = random.randint(5, world_generator.config.width - 5)
            y = random.randint(5, world_generator.config.height - 5)
            
            # Check if position is suitable
            if not world_generator.is_water(x, y):
                agent = RDR2Agent(x, y, i, role)
                agents.append(agent)
                break
            
            attempts += 1
        
        if attempts >= 50:
            # Fallback to any non-water position
            x = random.randint(5, world_generator.config.width - 5)
            y = random.randint(5, world_generator.config.height - 5)
            agent = RDR2Agent(x, y, i, role)
            agents.append(agent)
    
    print(f"Created {len(agents)} agents with role distribution:")
    for role in AgentRole:
        count = sum(1 for agent in agents if agent.role == role)
        print(f"  {role.value}: {count} ({count/len(agents)*100:.1f}%)")
    
    return agents 
#!/usr/bin/env python3
"""
Enhanced RDR2 Bio-Settlement Simulation with Physarum-Inspired Pathfinding
"""

import argparse
import sys
import random
import numpy as np
import time
import math
from typing import List, Tuple, Dict

def check_gpu_availability():
    """Check if GPU acceleration is available"""
    try:
        import cupy as cp
        
        # Test basic CuPy operation
        test_array = cp.array([1, 2, 3])
        result = cp.sum(test_array)
        
        # Get GPU info
        device = cp.cuda.Device()
        meminfo = device.mem_info
        free_mem = meminfo[0] / (1024**3)  # Convert to GB
        total_mem = meminfo[1] / (1024**3)
        
        return True, f"GPU available: {free_mem:.1f}GB free / {total_mem:.1f}GB total"
    
    except ImportError as e:
        return False, f"Missing GPU libraries: {e}"
    except Exception as e:
        return False, f"GPU test failed: {e}"

def create_agents(num_agents: int, world_generator) -> List:
    """Create RDR2 agents with role distribution"""
    from rdr2_agent import RDR2Agent, AgentRole
    
    agents = []
    
    # Role distribution (realistic proportions)
    role_distribution = {
        AgentRole.EXPLORER: 0.15,    # 15% explorers
        AgentRole.TRADER: 0.10,      # 10% traders  
        AgentRole.SETTLER: 0.22,     # 22% settlers
        AgentRole.MINER: 0.12,       # 12% miners
        AgentRole.FARMER: 0.23,      # 23% farmers
        AgentRole.HUNTER: 0.11,      # 11% hunters
        AgentRole.OUTLAW: 0.05,      # 5% outlaws
        AgentRole.LAWMAN: 0.02,      # 2% lawmen
    }
    
    # Create agents with role distribution
    role_counts = {}
    for role, percentage in role_distribution.items():
        count = int(num_agents * percentage)
        role_counts[role] = count
    
    # Fill remaining agents randomly
    total_assigned = sum(role_counts.values())
    remaining = num_agents - total_assigned
    
    # Distribute remaining agents
    roles = list(AgentRole)
    for _ in range(remaining):
        role = random.choice(roles)
        role_counts[role] = role_counts.get(role, 0) + 1
    
    agent_id = 0
    for role, count in role_counts.items():
        for _ in range(count):
            # Spawn near cities or random locations
            if world_generator.cities and random.random() < 0.6:
                # 60% spawn near cities
                city = random.choice(world_generator.cities)
                x = city.x + random.randint(-20, 20)
                y = city.y + random.randint(-20, 20)
                
                # Clamp to bounds
                x = max(5, min(world_generator.config.width - 5, x))
                y = max(5, min(world_generator.config.height - 5, y))
            else:
                # 40% spawn randomly
                x = random.randint(20, world_generator.config.width - 20)
                y = random.randint(20, world_generator.config.height - 20)
            
            agent = RDR2Agent(x, y, agent_id, role)
            agents.append(agent)
            agent_id += 1
    
    return agents

def run_cpu_visualization(world_generator, agents, trail_map):
    """Enhanced CPU simulation with professional visualization"""
    try:
        import pygame
        import numpy as np
        from rdr2_agent import AgentRole, PhysarumTrailSystem
        
        print("Starting Enhanced CPU Visualization with Physarum Trails...")
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Window setup
        WINDOW_WIDTH = 1400
        WINDOW_HEIGHT = 1000
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RDR2 Bio-Settlement Simulation - Physarum Trail System")
        clock = pygame.time.Clock()
        
        # Fonts
        font_large = pygame.font.Font(None, 32)
        font_medium = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 18)
        
        # Initialize physarum trail system
        trail_system = PhysarumTrailSystem(world_generator.config.width, world_generator.config.height)
        trail_system.set_city_attraction(world_generator.cities)
        
        # Map rendering setup
        MAP_WIDTH = 900
        MAP_HEIGHT = 700
        MAP_X = 50
        MAP_Y = 50
        
        # Scale factors
        scale_x = MAP_WIDTH / world_generator.config.width
        scale_y = MAP_HEIGHT / world_generator.config.height
        
        # UI Panel
        PANEL_X = MAP_X + MAP_WIDTH + 20
        PANEL_WIDTH = WINDOW_WIDTH - PANEL_X - 20
        
        # Colors
        BACKGROUND = (25, 25, 35)
        UI_PANEL = (40, 40, 50)
        UI_BORDER = (70, 70, 80)
        TEXT_WHITE = (255, 255, 255)
        TEXT_YELLOW = (255, 255, 100)
        TEXT_GREEN = (100, 255, 100)
        TEXT_RED = (255, 100, 100)
        TRAIL_COLOR_WEAK = (50, 100, 255, 40)    # Blue trails (weak)
        TRAIL_COLOR_MEDIUM = (100, 255, 100, 80) # Green trails (medium)
        TRAIL_COLOR_STRONG = (255, 255, 50, 120) # Yellow trails (strong)
        TRAIL_COLOR_HIGHWAY = (255, 100, 100, 160) # Red trails (highways)
        
        # Role colors
        role_colors = {
            AgentRole.EXPLORER: (255, 180, 50),    # Orange
            AgentRole.TRADER: (50, 255, 50),       # Green
            AgentRole.SETTLER: (100, 150, 255),    # Blue
            AgentRole.MINER: (150, 75, 0),         # Brown
            AgentRole.FARMER: (255, 255, 100),     # Yellow
            AgentRole.HUNTER: (0, 150, 0),         # Dark Green
            AgentRole.OUTLAW: (255, 50, 50),       # Red
            AgentRole.LAWMAN: (100, 100, 255),     # Light Blue
        }
        
        # Simulation state
        paused = False
        show_terrain = True
        show_agents = True
        show_cities = True
        show_trails = True
        trail_intensity = 1.0
        selected_role_filter = None
        camera_x, camera_y = 0, 0
        simulation_speed = 1
        frame_count = 0
        
        # Performance tracking
        update_times = []
        render_times = []
        
        # Statistics
        stats = {
            'total_agents': len(agents),
            'active_trails': 0,
            'trade_routes': 0,
            'settlements_connected': 0
        }
        
        print(f"✅ Simulation initialized with {len(agents)} agents")
        print(f"✅ Physarum trail system active")
        print(f"✅ {len(world_generator.cities)} cities ready for route formation")
        
        running = True
        while running:
            frame_start = time.time()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_t:
                        show_terrain = not show_terrain
                    elif event.key == pygame.K_a:
                        show_agents = not show_agents
                    elif event.key == pygame.K_c:
                        show_cities = not show_cities
                    elif event.key == pygame.K_r:
                        show_trails = not show_trails
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1:
                        selected_role_filter = AgentRole.EXPLORER if selected_role_filter != AgentRole.EXPLORER else None
                    elif event.key == pygame.K_2:
                        selected_role_filter = AgentRole.TRADER if selected_role_filter != AgentRole.TRADER else None
                    elif event.key == pygame.K_3:
                        selected_role_filter = AgentRole.SETTLER if selected_role_filter != AgentRole.SETTLER else None
                    elif event.key == pygame.K_4:
                        selected_role_filter = AgentRole.MINER if selected_role_filter != AgentRole.MINER else None
                    elif event.key == pygame.K_5:
                        selected_role_filter = AgentRole.FARMER if selected_role_filter != AgentRole.FARMER else None
                    elif event.key == pygame.K_6:
                        selected_role_filter = AgentRole.HUNTER if selected_role_filter != AgentRole.HUNTER else None
                    elif event.key == pygame.K_7:
                        selected_role_filter = AgentRole.OUTLAW if selected_role_filter != AgentRole.OUTLAW else None
                    elif event.key == pygame.K_8:
                        selected_role_filter = AgentRole.LAWMAN if selected_role_filter != AgentRole.LAWMAN else None
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                        trail_intensity = min(3.0, trail_intensity + 0.2)
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        trail_intensity = max(0.2, trail_intensity - 0.2)
            
            # Update simulation
            if not paused:
                update_start = time.time()
                
                # Update trail system
                trail_system.update_trails()
                
                # Update agents with physarum pathfinding
                for agent in agents:
                    if frame_count % simulation_speed == 0:  # Control simulation speed
                        agent.update(world_generator, agents, trail_system)
                
                update_times.append(time.time() - update_start)
                if len(update_times) > 60:
                    update_times.pop(0)
            
            # Render
            render_start = time.time()
            screen.fill(BACKGROUND)
            
            # Draw map border
            pygame.draw.rect(screen, UI_BORDER, (MAP_X-2, MAP_Y-2, MAP_WIDTH+4, MAP_HEIGHT+4), 2)
            
            # Create map surface for better performance
            map_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
            
            # Draw terrain
            if show_terrain:
                for y in range(0, world_generator.config.height, 2):  # Skip every other pixel for performance
                    for x in range(0, world_generator.config.width, 2):
                        terrain_type = world_generator.terrain[y, x]
                        color = world_generator.get_terrain_color(terrain_type)
                        
                        # Convert world coordinates to screen coordinates
                        screen_x = int(x * scale_x)
                        screen_y = int(y * scale_y)
                        
                        if 0 <= screen_x < MAP_WIDTH and 0 <= screen_y < MAP_HEIGHT:
                            pygame.draw.rect(map_surface, color, (screen_x, screen_y, max(1, int(scale_x)), max(1, int(scale_y))))
            
            # Draw physarum trails (this is the key feature!)
            if show_trails:
                trail_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)
                
                max_trail = np.max(trail_system.trail_map) if np.max(trail_system.trail_map) > 0 else 1
                
                for y in range(0, world_generator.config.height, 1):
                    for x in range(0, world_generator.config.width, 1):
                        trail_strength = trail_system.trail_map[y, x]
                        
                        if trail_strength > 0.1:  # Only draw visible trails
                            # Convert to screen coordinates
                            screen_x = int(x * scale_x)
                            screen_y = int(y * scale_y)
                            
                            if 0 <= screen_x < MAP_WIDTH and 0 <= screen_y < MAP_HEIGHT:
                                # Determine trail color based on strength
                                normalized_strength = min(1.0, trail_strength / max_trail)
                                alpha = int(255 * normalized_strength * trail_intensity)
                                
                                if trail_strength < 2.0:
                                    color = (*TRAIL_COLOR_WEAK[:3], min(255, alpha))
                                elif trail_strength < 5.0:
                                    color = (*TRAIL_COLOR_MEDIUM[:3], min(255, alpha))
                                elif trail_strength < 10.0:
                                    color = (*TRAIL_COLOR_STRONG[:3], min(255, alpha))
                                else:
                                    color = (*TRAIL_COLOR_HIGHWAY[:3], min(255, alpha))
                                
                                # Draw trail pixel
                                size = max(1, int(scale_x * (1 + normalized_strength)))
                                pygame.draw.circle(trail_surface, color, (screen_x, screen_y), size)
                
                map_surface.blit(trail_surface, (0, 0))
            
            # Draw cities
            if show_cities:
                for city in world_generator.cities:
                    screen_x = int(city.x * scale_x)
                    screen_y = int(city.y * scale_y)
                    
                    # City circle
                    pygame.draw.circle(map_surface, (255, 255, 255), (screen_x, screen_y), 8)
                    pygame.draw.circle(map_surface, (200, 200, 0), (screen_x, screen_y), 6)
                    
                    # City name (smaller font for performance)
                    if MAP_WIDTH > 600:  # Only show names if map is large enough
                        name_surface = font_small.render(city.name, True, TEXT_WHITE)
                        map_surface.blit(name_surface, (screen_x + 10, screen_y - 8))
            
            # Draw agents
            if show_agents:
                agent_counts = {}
                for agent in agents:
                    if selected_role_filter is None or agent.role == selected_role_filter:
                        screen_x = int(agent.x * scale_x)
                        screen_y = int(agent.y * scale_y)
                        
                        if 0 <= screen_x < MAP_WIDTH and 0 <= screen_y < MAP_HEIGHT:
                            color = role_colors.get(agent.role, (255, 255, 255))
                            pygame.draw.circle(map_surface, color, (screen_x, screen_y), 3)
                            
                            # Count agents by role
                            agent_counts[agent.role] = agent_counts.get(agent.role, 0) + 1
            
            # Blit map surface to screen
            screen.blit(map_surface, (MAP_X, MAP_Y))
            
            # Draw UI Panel
            pygame.draw.rect(screen, UI_PANEL, (PANEL_X, MAP_Y, PANEL_WIDTH, MAP_HEIGHT))
            pygame.draw.rect(screen, UI_BORDER, (PANEL_X, MAP_Y, PANEL_WIDTH, MAP_HEIGHT), 2)
            
            # UI Content
            y_offset = MAP_Y + 20
            
            # Title
            title = font_large.render("RDR2 Physarum Simulation", True, TEXT_YELLOW)
            screen.blit(title, (PANEL_X + 10, y_offset))
            y_offset += 40
            
            # Status
            status_text = "RUNNING" if not paused else "PAUSED"
            status_color = TEXT_GREEN if not paused else TEXT_RED
            status = font_medium.render(f"Status: {status_text}", True, status_color)
            screen.blit(status, (PANEL_X + 10, y_offset))
            y_offset += 30
            
            # Trail Statistics
            active_trails = np.sum(trail_system.trail_map > 0.5)
            strong_trails = np.sum(trail_system.trail_map > 5.0)
            highway_trails = np.sum(trail_system.trail_map > 10.0)
            
            trail_stats = [
                f"Active Trails: {int(active_trails)}",
                f"Major Routes: {int(strong_trails)}",
                f"Trade Highways: {int(highway_trails)}",
                f"Trail Intensity: {trail_intensity:.1f}x"
            ]
            
            for stat in trail_stats:
                text = font_small.render(stat, True, TEXT_WHITE)
                screen.blit(text, (PANEL_X + 10, y_offset))
                y_offset += 20
            
            y_offset += 10
            
            # Agent counts by role
            role_header = font_medium.render("Agent Distribution:", True, TEXT_YELLOW)
            screen.blit(role_header, (PANEL_X + 10, y_offset))
            y_offset += 25
            
            for role in AgentRole:
                count = len([a for a in agents if a.role == role])
                percentage = (count / len(agents)) * 100
                color = role_colors.get(role, TEXT_WHITE)
                
                # Role name and count
                role_text = font_small.render(f"{role.value.title()}: {count} ({percentage:.1f}%)", True, color)
                screen.blit(role_text, (PANEL_X + 10, y_offset))
                y_offset += 18
            
            y_offset += 20
            
            # Controls
            controls_header = font_medium.render("Controls:", True, TEXT_YELLOW)
            screen.blit(controls_header, (PANEL_X + 10, y_offset))
            y_offset += 25
            
            controls = [
                "SPACE - Pause/Resume",
                "T - Toggle terrain",
                "A - Toggle agents", 
                "C - Toggle cities",
                "R - Toggle trails",
                "+/- - Trail intensity",
                "1-8 - Filter by role",
                "ESC - Exit"
            ]
            
            for control in controls:
                text = font_small.render(control, True, TEXT_WHITE)
                screen.blit(text, (PANEL_X + 10, y_offset))
                y_offset += 16
            
            # Performance info
            if update_times:
                avg_update = np.mean(update_times) * 1000
                fps = clock.get_fps()
                
                y_offset += 20
                perf_header = font_medium.render("Performance:", True, TEXT_YELLOW)
                screen.blit(perf_header, (PANEL_X + 10, y_offset))
                y_offset += 25
                
                perf_stats = [
                    f"FPS: {fps:.1f}",
                    f"Update: {avg_update:.1f}ms",
                    f"Agents: {len(agents)}"
                ]
                
                for stat in perf_stats:
                    text = font_small.render(stat, True, TEXT_WHITE)
                    screen.blit(text, (PANEL_X + 10, y_offset))
                    y_offset += 16
            
            render_times.append(time.time() - render_start)
            if len(render_times) > 60:
                render_times.pop(0)
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS target
            
            frame_count += 1
            
        pygame.quit()
        
    except ImportError as e:
        print(f"❌ Error: Missing required library: {e}")
        print("Please install pygame: pip install pygame")
        return False
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return False
    
    return True

def run_gpu_simulation(world_generator, agents, trail_map):
    """Run GPU-accelerated simulation"""
    try:
        from rdr2_gpu_simulation import RDR2GPUSimulation
        from rdr2_agent import PhysarumTrailSystem
        
        print("🚀 Starting GPU-accelerated simulation with Physarum trails...")
        
        # Initialize trail system
        trail_system = PhysarumTrailSystem(world_generator.config.width, world_generator.config.height)
        trail_system.set_city_attraction(world_generator.cities)
        
        # Create GPU simulation
        gpu_sim = RDR2GPUSimulation(world_generator, agents, trail_system)
        
        # Run simulation
        return gpu_sim.run()
        
    except ImportError as e:
        print(f"❌ GPU libraries not available: {e}")
        print("💡 Falling back to CPU simulation...")
        return run_cpu_visualization(world_generator, agents, trail_map)
    except Exception as e:
        print(f"❌ GPU simulation failed: {e}")
        print("💡 Falling back to CPU simulation...")
        return run_cpu_visualization(world_generator, agents, trail_map)

def main():
    parser = argparse.ArgumentParser(description="RDR2 Bio-Settlement Simulation with Physarum Pathfinding")
    parser.add_argument("--agents", type=int, default=500, help="Number of agents (default: 500)")
    parser.add_argument("--width", type=int, default=400, help="World width (default: 400)")
    parser.add_argument("--height", type=int, default=300, help="World height (default: 300)")
    parser.add_argument("--cities", type=int, default=7, help="Number of cities (default: 7)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only simulation")
    parser.add_argument("--gpu", action="store_true", help="Force GPU simulation")
    parser.add_argument("--test-gpu", action="store_true", help="Test GPU availability")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")
    
    args = parser.parse_args()
    
    # Test GPU if requested
    if args.test_gpu:
        gpu_available, gpu_info = check_gpu_availability()
        print(f"GPU Test: {gpu_info}")
        return
    
    # Set random seed
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    print("🎮 RDR2 Bio-Settlement Simulation with Physarum Intelligence")
    print("=" * 65)
    
    # Import world generator
    try:
        from rdr2_world_generator import RDR2WorldGenerator, RDR2WorldConfig
    except ImportError as e:
        print(f"❌ Error importing world generator: {e}")
        return
    
    # Create world
    print("🗺️  Generating world...")
    config = RDR2WorldConfig(
        width=args.width,
        height=args.height,
        num_cities=args.cities
    )
    
    world_generator = RDR2WorldGenerator(config)
    world_generator.generate_world()
    
    print(f"✅ World generated: {len(world_generator.cities)} cities")
    for city in world_generator.cities:
        print(f"   🏛️  {city.name} (pop: {city.population:,}) at ({city.x}, {city.y})")
    
    # Create agents
    print(f"\n👥 Creating {args.agents} agents...")
    agents = create_agents(args.agents, world_generator)
    
    # Count agents by role
    from rdr2_agent import AgentRole
    role_counts = {}
    for agent in agents:
        role_counts[agent.role] = role_counts.get(agent.role, 0) + 1
    
    print("✅ Agents created:")
    for role, count in role_counts.items():
        percentage = (count / len(agents)) * 100
        print(f"   {role.value.title()}: {count} ({percentage:.1f}%)")
    
    # Initialize trail map
    trail_map = np.zeros((config.height, config.width), dtype=np.float32)
    
    # Determine simulation type
    if args.cpu:
        print("\n🖥️  Running CPU simulation...")
        success = run_cpu_visualization(world_generator, agents, trail_map)
    elif args.gpu:
        print("\n🚀 Testing GPU availability...")
        gpu_available, gpu_info = check_gpu_availability()
        print(f"GPU Status: {gpu_info}")
        
        if gpu_available:
            success = run_gpu_simulation(world_generator, agents, trail_map)
        else:
            print("💡 Falling back to CPU simulation...")
            success = run_cpu_visualization(world_generator, agents, trail_map)
    else:
        # Auto-detect
        print("\n🔍 Auto-detecting best simulation mode...")
        gpu_available, gpu_info = check_gpu_availability()
        print(f"GPU Status: {gpu_info}")
        
        if gpu_available and args.agents >= 100:
            print("🚀 Using GPU acceleration for optimal performance...")
            success = run_gpu_simulation(world_generator, agents, trail_map)
        else:
            print("🖥️  Using CPU simulation...")
            success = run_cpu_visualization(world_generator, agents, trail_map)
    
    if success:
        print("\n✅ Simulation completed successfully!")
        print("\n🧠 Physarum Trail System Features:")
        print("   • Intelligent pathfinding between cities")
        print("   • Dynamic trade route formation")
        print("   • Role-based trail behaviors")
        print("   • Emergent transportation networks")
        print("\n🎯 Watch for:")
        print("   • Blue trails: Exploration paths")
        print("   • Green trails: Established routes")
        print("   • Yellow trails: Major trade routes")
        print("   • Red trails: Transportation highways")
    else:
        print("\n❌ Simulation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
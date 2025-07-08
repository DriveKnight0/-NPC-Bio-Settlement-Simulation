import pygame
import numpy as np
import math
import colorsys
from typing import List, Tuple, Dict, Optional
import time

from rdr2_world_generator import RDR2WorldGenerator, TERRAIN_COLORS
from rdr2_agent import RDR2Agent, AgentRole
from rdr2_gpu_simulation import RDR2GPUSimulation

class RDR2Visualizer:
    def __init__(self, simulation: RDR2GPUSimulation, window_width: int = 1200, window_height: int = 900):
        self.simulation = simulation
        self.world_generator = simulation.get_world_generator()
        
        # Display settings
        self.window_width = window_width
        self.window_height = window_height
        self.map_width = window_width - 300  # Leave space for UI
        self.map_height = window_height - 100
        
        # Calculate scale factors
        self.scale_x = self.map_width / simulation.config.width
        self.scale_y = self.map_height / simulation.config.height
        self.scale = min(self.scale_x, self.scale_y)
        
        # Visualization settings
        self.show_terrain = True
        self.show_trails = True
        self.show_agents = True
        self.show_cities = True
        self.show_roads = True
        self.show_railways = True
        self.show_agent_info = False
        self.show_statistics = True
        
        # Camera/view settings
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        # Agent filtering
        self.filter_roles = set(AgentRole)
        
        # Colors and visual settings
        self.trail_alpha = 0.6
        self.agent_size = max(2, int(self.scale * 0.8))
        
        # UI fonts
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("RDR2-Inspired Bio Settlement Simulation")
        
        # Create surfaces for different layers
        self.terrain_surface = pygame.Surface((self.map_width, self.map_height))
        self.trail_surface = pygame.Surface((self.map_width, self.map_height))
        self.trail_surface.set_alpha(int(255 * self.trail_alpha))
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # UI state
        self.paused = False
        self.selected_agent = None
        self.mouse_x = 0
        self.mouse_y = 0
        
        print(f"Visualizer initialized:")
        print(f"  Window: {window_width}x{window_height}")
        print(f"  Map area: {self.map_width}x{self.map_height}")
        print(f"  Scale: {self.scale:.2f}")
        print(f"  Agent size: {self.agent_size}")
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int((world_x - self.camera_x) * self.scale * self.zoom)
        screen_y = int((world_y - self.camera_y) * self.scale * self.zoom)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        world_x = screen_x / (self.scale * self.zoom) + self.camera_x
        world_y = screen_y / (self.scale * self.zoom) + self.camera_y
        return world_x, world_y
    
    def update(self):
        """Update visualization"""
        # Handle events
        self._handle_events()
        
        # Update simulation if not paused
        if not self.paused:
            self.simulation.update()
        
        # Get visualization data
        terrain, trails, agents = self.simulation.get_visualization_data()
        
        # Render everything
        self._render_terrain(terrain)
        self._render_trails(trails)
        self._render_infrastructure()
        self._render_cities()
        self._render_agents(agents)
        self._render_ui()
        
        # Update display
        pygame.display.flip()
        
        # Update FPS counter
        self._update_fps()
    
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event)
            
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)
            
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_x, self.mouse_y = event.pos
        
        return True
    
    def _handle_keydown(self, event):
        """Handle keyboard input"""
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_t:
            self.show_terrain = not self.show_terrain
        elif event.key == pygame.K_r:
            self.show_trails = not self.show_trails
        elif event.key == pygame.K_a:
            self.show_agents = not self.show_agents
        elif event.key == pygame.K_c:
            self.show_cities = not self.show_cities
        elif event.key == pygame.K_o:
            self.show_roads = not self.show_roads
        elif event.key == pygame.K_l:
            self.show_railways = not self.show_railways
        elif event.key == pygame.K_i:
            self.show_agent_info = not self.show_agent_info
        elif event.key == pygame.K_s:
            self.show_statistics = not self.show_statistics
        elif event.key == pygame.K_ESCAPE:
            self.selected_agent = None
        
        # Camera movement
        elif event.key == pygame.K_w:
            self.camera_y -= 10 / self.zoom
        elif event.key == pygame.K_s:
            self.camera_y += 10 / self.zoom
        elif event.key == pygame.K_a:
            self.camera_x -= 10 / self.zoom
        elif event.key == pygame.K_d:
            self.camera_x += 10 / self.zoom
        
        # Zoom
        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
            self.zoom = min(self.max_zoom, self.zoom * 1.2)
        elif event.key == pygame.K_MINUS:
            self.zoom = max(self.min_zoom, self.zoom / 1.2)
        
        # Agent role filtering (number keys)
        elif event.key >= pygame.K_1 and event.key <= pygame.K_8:
            role_index = event.key - pygame.K_1
            roles = list(AgentRole)
            if role_index < len(roles):
                role = roles[role_index]
                if role in self.filter_roles:
                    self.filter_roles.remove(role)
                else:
                    self.filter_roles.add(role)
    
    def _handle_mouse_click(self, event):
        """Handle mouse clicks"""
        if event.button == 1:  # Left click
            if event.pos[0] < self.map_width and event.pos[1] < self.map_height:
                # Click on map - select agent
                world_x, world_y = self.screen_to_world(event.pos[0], event.pos[1])
                self._select_agent_at(world_x, world_y)
        
        elif event.button == 3:  # Right click
            # Pan to location
            if event.pos[0] < self.map_width and event.pos[1] < self.map_height:
                world_x, world_y = self.screen_to_world(event.pos[0], event.pos[1])
                self.camera_x = world_x - (self.map_width / 2) / (self.scale * self.zoom)
                self.camera_y = world_y - (self.map_height / 2) / (self.scale * self.zoom)
    
    def _handle_mouse_wheel(self, event):
        """Handle mouse wheel zoom"""
        if self.mouse_x < self.map_width and self.mouse_y < self.map_height:
            # Zoom towards mouse position
            old_zoom = self.zoom
            
            if event.y > 0:
                self.zoom = min(self.max_zoom, self.zoom * 1.1)
            else:
                self.zoom = max(self.min_zoom, self.zoom / 1.1)
            
            # Adjust camera to zoom towards mouse
            if old_zoom != self.zoom:
                world_x, world_y = self.screen_to_world(self.mouse_x, self.mouse_y)
                self.camera_x = world_x - self.mouse_x / (self.scale * self.zoom)
                self.camera_y = world_y - self.mouse_y / (self.scale * self.zoom)
    
    def _select_agent_at(self, world_x: float, world_y: float):
        """Select agent at world coordinates"""
        _, _, agents = self.simulation.get_visualization_data()
        
        min_distance = float('inf')
        closest_agent = None
        
        for agent in agents:
            distance = math.sqrt((agent.x - world_x)**2 + (agent.y - world_y)**2)
            if distance < 3.0 and distance < min_distance:
                min_distance = distance
                closest_agent = agent
        
        self.selected_agent = closest_agent
    
    def _render_terrain(self, terrain: np.ndarray):
        """Render terrain layer"""
        if not self.show_terrain:
            return
        
        self.terrain_surface.fill((0, 0, 0))
        
        # Calculate visible area
        visible_x1 = max(0, int(self.camera_x))
        visible_y1 = max(0, int(self.camera_y))
        visible_x2 = min(self.simulation.config.width, 
                        int(self.camera_x + self.map_width / (self.scale * self.zoom)) + 1)
        visible_y2 = min(self.simulation.config.height, 
                        int(self.camera_y + self.map_height / (self.scale * self.zoom)) + 1)
        
        # Render terrain tiles
        for y in range(visible_y1, visible_y2):
            for x in range(visible_x1, visible_x2):
                terrain_type = terrain[y, x]
                color = TERRAIN_COLORS.get(terrain_type, (128, 128, 128))
                
                screen_x, screen_y = self.world_to_screen(x, y)
                
                # Only draw if on screen
                if (0 <= screen_x < self.map_width and 0 <= screen_y < self.map_height):
                    tile_size = max(1, int(self.scale * self.zoom))
                    pygame.draw.rect(self.terrain_surface, color, 
                                   (screen_x, screen_y, tile_size, tile_size))
        
        self.screen.blit(self.terrain_surface, (0, 0))
    
    def _render_trails(self, trails: np.ndarray):
        """Render pheromone trails"""
        if not self.show_trails:
            return
        
        self.trail_surface.fill((0, 0, 0, 0))
        
        # Calculate visible area
        visible_x1 = max(0, int(self.camera_x))
        visible_y1 = max(0, int(self.camera_y))
        visible_x2 = min(self.simulation.config.width, 
                        int(self.camera_x + self.map_width / (self.scale * self.zoom)) + 1)
        visible_y2 = min(self.simulation.config.height, 
                        int(self.camera_y + self.map_height / (self.scale * self.zoom)) + 1)
        
        # Find max trail value for normalization
        max_trail = np.max(trails) if np.max(trails) > 0 else 1.0
        
        # Render trails
        for y in range(visible_y1, visible_y2):
            for x in range(visible_x1, visible_x2):
                trail_value = trails[y, x]
                
                if trail_value > 0.1:
                    # Normalize trail value
                    intensity = min(1.0, trail_value / max_trail)
                    
                    # Create heat map color (blue -> green -> yellow -> red)
                    hue = (1.0 - intensity) * 0.7  # Blue to red
                    rgb = colorsys.hsv_to_rgb(hue, 1.0, intensity)
                    color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                    
                    screen_x, screen_y = self.world_to_screen(x, y)
                    
                    if (0 <= screen_x < self.map_width and 0 <= screen_y < self.map_height):
                        tile_size = max(1, int(self.scale * self.zoom))
                        pygame.draw.rect(self.trail_surface, color, 
                                       (screen_x, screen_y, tile_size, tile_size))
        
        self.screen.blit(self.trail_surface, (0, 0))
    
    def _render_infrastructure(self):
        """Render roads and railways"""
        if not (self.show_roads or self.show_railways):
            return
        
        # Render roads
        if self.show_roads:
            for road_x, road_y in self.world_generator.roads:
                screen_x, screen_y = self.world_to_screen(road_x, road_y)
                
                if (0 <= screen_x < self.map_width and 0 <= screen_y < self.map_height):
                    road_size = max(1, int(self.scale * self.zoom * 0.8))
                    pygame.draw.circle(self.screen, (139, 69, 19), (screen_x, screen_y), road_size)
        
        # Render railways
        if self.show_railways:
            for rail_x, rail_y in self.world_generator.railways:
                screen_x, screen_y = self.world_to_screen(rail_x, rail_y)
                
                if (0 <= screen_x < self.map_width and 0 <= screen_y < self.map_height):
                    rail_size = max(1, int(self.scale * self.zoom * 0.6))
                    pygame.draw.circle(self.screen, (64, 64, 64), (screen_x, screen_y), rail_size)
    
    def _render_cities(self):
        """Render cities and settlements"""
        if not self.show_cities:
            return
        
        for city in self.world_generator.cities:
            screen_x, screen_y = self.world_to_screen(city.x, city.y)
            
            if (0 <= screen_x < self.map_width and 0 <= screen_y < self.map_height):
                # City size based on population
                if city.population > 3000:
                    size = max(6, int(self.scale * self.zoom * 2.5))
                    color = (255, 255, 255)
                elif city.population > 1000:
                    size = max(4, int(self.scale * self.zoom * 2.0))
                    color = (220, 220, 220)
                else:
                    size = max(3, int(self.scale * self.zoom * 1.5))
                    color = (180, 180, 180)
                
                # Draw city
                pygame.draw.circle(self.screen, (0, 0, 0), (screen_x, screen_y), size + 1)
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), size)
                
                # Draw city name if zoomed in enough
                if self.zoom > 1.0:
                    text = self.font_small.render(city.name, True, (255, 255, 255))
                    text_rect = text.get_rect()
                    text_rect.centerx = screen_x
                    text_rect.top = screen_y + size + 2
                    
                    # Background for text
                    pygame.draw.rect(self.screen, (0, 0, 0, 128), text_rect.inflate(4, 2))
                    self.screen.blit(text, text_rect)
    
    def _render_agents(self, agents: List[RDR2Agent]):
        """Render agents"""
        if not self.show_agents:
            return
        
        agent_count = 0
        
        for agent in agents:
            # Filter by role
            if agent.role not in self.filter_roles:
                continue
            
            screen_x, screen_y = self.world_to_screen(agent.x, agent.y)
            
            # Only render agents on screen
            if (0 <= screen_x < self.map_width and 0 <= screen_y < self.map_height):
                agent_count += 1
                
                # Get agent color
                color = agent.get_role_color()
                
                # Highlight selected agent
                if agent == self.selected_agent:
                    size = self.agent_size + 2
                    pygame.draw.circle(self.screen, (255, 255, 0), (screen_x, screen_y), size)
                
                # Draw agent
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), self.agent_size)
                
                # Draw target if selected
                if agent == self.selected_agent and agent.target_x is not None:
                    target_screen_x, target_screen_y = self.world_to_screen(agent.target_x, agent.target_y)
                    if (0 <= target_screen_x < self.map_width and 0 <= target_screen_y < self.map_height):
                        pygame.draw.line(self.screen, (255, 255, 0), 
                                       (screen_x, screen_y), (target_screen_x, target_screen_y), 1)
                        pygame.draw.circle(self.screen, (255, 255, 0), 
                                         (target_screen_x, target_screen_y), 3)
    
    def _render_ui(self):
        """Render user interface"""
        # UI background
        ui_rect = pygame.Rect(self.map_width, 0, 300, self.window_height)
        pygame.draw.rect(self.screen, (40, 40, 40), ui_rect)
        pygame.draw.line(self.screen, (80, 80, 80), (self.map_width, 0), (self.map_width, self.window_height), 2)
        
        y_offset = 10
        
        # Title
        title = self.font_large.render("RDR2 Bio-Sim", True, (255, 255, 255))
        self.screen.blit(title, (self.map_width + 10, y_offset))
        y_offset += 35
        
        # FPS and status
        fps_text = f"FPS: {self.current_fps:.1f}"
        if self.paused:
            fps_text += " (PAUSED)"
        
        fps_surface = self.font_medium.render(fps_text, True, (255, 255, 255))
        self.screen.blit(fps_surface, (self.map_width + 10, y_offset))
        y_offset += 25
        
        # Statistics
        if self.show_statistics:
            stats = self.simulation.get_statistics()
            
            stat_texts = [
                f"Frame: {stats['frame']}",
                f"Agents: {stats['agents']}",
                f"Cities: {stats['cities']}",
                f"World: {stats['world_size']}",
                f"Coverage: {stats['coverage']:.2%}",
                f"Trail Sum: {stats['trail_sum']:.1f}"
            ]
            
            for text in stat_texts:
                surface = self.font_small.render(text, True, (200, 200, 200))
                self.screen.blit(surface, (self.map_width + 10, y_offset))
                y_offset += 18
        
        y_offset += 10
        
        # Role distribution
        role_header = self.font_medium.render("Agent Roles:", True, (255, 255, 255))
        self.screen.blit(role_header, (self.map_width + 10, y_offset))
        y_offset += 25
        
        stats = self.simulation.get_statistics()
        role_dist = stats.get('role_distribution', {})
        
        for i, role in enumerate(AgentRole):
            count = role_dist.get(role.value, 0)
            color = (255, 255, 255) if role in self.filter_roles else (100, 100, 100)
            
            role_text = f"{i+1}. {role.value}: {count}"
            surface = self.font_small.render(role_text, True, color)
            self.screen.blit(surface, (self.map_width + 10, y_offset))
            
            # Role color indicator
            role_color = self.simulation.agents[0].get_role_color() if self.simulation.agents else (255, 255, 255)
            for agent in self.simulation.agents:
                if agent.role == role:
                    role_color = agent.get_role_color()
                    break
            
            pygame.draw.circle(self.screen, role_color, 
                             (self.map_width + 260, y_offset + 8), 6)
            
            y_offset += 18
        
        y_offset += 10
        
        # Selected agent info
        if self.selected_agent and self.show_agent_info:
            agent_header = self.font_medium.render("Selected Agent:", True, (255, 255, 255))
            self.screen.blit(agent_header, (self.map_width + 10, y_offset))
            y_offset += 25
            
            agent_info = [
                f"ID: {self.selected_agent.id}",
                f"Role: {self.selected_agent.role.value}",
                f"State: {self.selected_agent.state.value}",
                f"Position: ({self.selected_agent.x}, {self.selected_agent.y})",
                f"Health: {self.selected_agent.stats.health:.1f}",
                f"Energy: {self.selected_agent.stats.energy:.1f}",
                f"Wealth: {self.selected_agent.stats.wealth:.1f}"
            ]
            
            if self.selected_agent.target_x is not None:
                agent_info.append(f"Target: ({self.selected_agent.target_x}, {self.selected_agent.target_y})")
            
            for info in agent_info:
                surface = self.font_small.render(info, True, (200, 200, 200))
                self.screen.blit(surface, (self.map_width + 10, y_offset))
                y_offset += 16
        
        # Controls
        y_offset = self.window_height - 200
        
        controls_header = self.font_medium.render("Controls:", True, (255, 255, 255))
        self.screen.blit(controls_header, (self.map_width + 10, y_offset))
        y_offset += 25
        
        controls = [
            "SPACE: Pause/Resume",
            "T: Toggle Terrain",
            "R: Toggle Trails", 
            "A: Toggle Agents",
            "C: Toggle Cities",
            "O: Toggle Roads",
            "L: Toggle Railways",
            "1-8: Filter Roles",
            "WASD: Move Camera",
            "+/-: Zoom",
            "Click: Select Agent",
            "Right Click: Pan"
        ]
        
        for control in controls:
            if y_offset < self.window_height - 10:
                surface = self.font_small.render(control, True, (150, 150, 150))
                self.screen.blit(surface, (self.map_width + 10, y_offset))
                y_offset += 14
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def run(self):
        """Run the visualization loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.update()
            if running is False:
                break
            
            clock.tick(60)  # 60 FPS cap
        
        pygame.quit()

def run_rdr2_visualization(num_agents: int = 1000):
    """Run RDR2 visualization with specified number of agents"""
    print(f"Starting RDR2 simulation with {num_agents} agents...")
    
    # Create simulation
    from rdr2_gpu_simulation import create_rdr2_gpu_simulation
    simulation = create_rdr2_gpu_simulation(num_agents)
    
    # Create and run visualizer
    visualizer = RDR2Visualizer(simulation)
    visualizer.run()

if __name__ == "__main__":
    run_rdr2_visualization(1000) 
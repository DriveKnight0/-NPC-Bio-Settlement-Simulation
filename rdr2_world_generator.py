import numpy as np
import random
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import euclidean
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import math

# Enhanced terrain types inspired by RDR2
WATER = 0
PLAINS = 1
FOREST = 2
MOUNTAIN = 3
DESERT = 4
SWAMP = 5
SNOW_MOUNTAIN = 6
FOOTHILLS = 7
ALPINE_MEADOW = 8
RIVER = 9
ROAD = 10
RAILWAY = 11
CITY = 12
TOWN = 13
SETTLEMENT = 14

# RDR2-inspired terrain colors
TERRAIN_COLORS = {
    WATER: (0, 100, 200),
    PLAINS: (120, 180, 80),
    FOREST: (34, 80, 34),
    MOUNTAIN: (100, 100, 100),
    DESERT: (220, 180, 120),
    SWAMP: (80, 120, 80),
    SNOW_MOUNTAIN: (240, 240, 255),
    FOOTHILLS: (140, 120, 90),
    ALPINE_MEADOW: (150, 200, 120),
    RIVER: (100, 150, 255),
    ROAD: (180, 160, 140),
    RAILWAY: (80, 60, 40),
    CITY: (200, 200, 200),
    TOWN: (170, 170, 170),
    SETTLEMENT: (140, 140, 140)
}

# Terrain movement costs (for pathfinding)
TERRAIN_COSTS = {
    WATER: 10.0,
    PLAINS: 1.0,
    FOREST: 2.0,
    MOUNTAIN: 5.0,
    DESERT: 2.5,
    SWAMP: 4.0,
    SNOW_MOUNTAIN: 8.0,
    FOOTHILLS: 3.0,
    ALPINE_MEADOW: 1.5,
    RIVER: 2.0,
    ROAD: 0.5,
    RAILWAY: 0.3,
    CITY: 1.0,
    TOWN: 1.0,
    SETTLEMENT: 1.0
}

@dataclass
class City:
    name: str
    x: int
    y: int
    population: int
    city_type: str
    terrain_preference: int

@dataclass
class RDR2WorldConfig:
    width: int = 250
    height: int = 250
    
    # Mountain configuration
    num_mountain_ranges: int = 3
    mountain_range_length: int = 40
    mountain_width: int = 15
    
    # Water and rivers
    num_major_rivers: int = 4
    num_minor_rivers: int = 8
    river_width: int = 2
    
    # Cities and settlements
    num_cities: int = 7
    num_towns: int = 15
    num_settlements: int = 25
    
    # Infrastructure
    road_density: float = 0.3
    railway_lines: int = 3
    
    # Terrain distribution
    forest_coverage: float = 0.25
    desert_coverage: float = 0.15
    swamp_coverage: float = 0.08
    plains_coverage: float = 0.35

class RDR2WorldGenerator:
    def __init__(self, config: RDR2WorldConfig):
        self.config = config
        self.terrain = np.zeros((config.height, config.width), dtype=int)
        self.elevation = np.zeros((config.height, config.width), dtype=float)
        self.cities: List[City] = []
        self.roads: Set[Tuple[int, int]] = set()
        self.railways: Set[Tuple[int, int]] = set()
        self.rivers: Set[Tuple[int, int]] = set()
        
        # City templates for demonstration
        self.city_templates = [
            {"name": "City A", "type": "industrial_port", "population": 8000, "terrain": SWAMP},
            {"name": "City B", "type": "livestock_town", "population": 1200, "terrain": PLAINS},
            {"name": "City C", "type": "logging_town", "population": 800, "terrain": FOREST},
            {"name": "City D", "type": "desert_town", "population": 600, "terrain": DESERT},
            {"name": "City E", "type": "mining_town", "population": 1500, "terrain": MOUNTAIN},
            {"name": "City F", "type": "agricultural_town", "population": 1000, "terrain": PLAINS},
            {"name": "City G", "type": "frontier_city", "population": 3000, "terrain": PLAINS}
        ]
    
    def generate_world(self) -> np.ndarray:
        """Generate the complete RDR2-inspired world"""
        print("Generating RDR2-inspired world...")
        
        # Step 1: Generate base elevation
        self._generate_elevation()
        
        # Step 2: Create mountain ranges
        self._create_mountain_ranges()
        
        # Step 3: Generate base terrain types
        self._generate_base_terrain()
        
        # Step 4: Create water bodies and rivers
        self._create_water_systems()
        
        # Step 5: Place cities strategically
        self._place_cities()
        
        # Step 6: Create road network
        self._create_road_network()
        
        # Step 7: Build railway system
        self._create_railway_system()
        
        # Step 8: Add final details
        self._add_terrain_details()
        
        print(f"World generated: {len(self.cities)} cities, {len(self.roads)} road tiles, {len(self.railways)} railway tiles")
        return self.terrain
    
    def _generate_elevation(self):
        """Generate realistic elevation map"""
        # Create multiple noise layers for realistic terrain
        noise1 = np.random.random((self.config.height, self.config.width))
        noise2 = np.random.random((self.config.height // 2, self.config.width // 2))
        noise3 = np.random.random((self.config.height // 4, self.config.width // 4))
        
        # Resize smaller noise maps to exact target size
        from scipy.ndimage import zoom
        zoom_factor_2 = (self.config.height / noise2.shape[0], self.config.width / noise2.shape[1])
        zoom_factor_3 = (self.config.height / noise3.shape[0], self.config.width / noise3.shape[1])
        
        noise2 = zoom(noise2, zoom_factor_2, order=1)
        noise3 = zoom(noise3, zoom_factor_3, order=1)
        
        # Ensure exact dimensions
        noise2 = noise2[:self.config.height, :self.config.width]
        noise3 = noise3[:self.config.height, :self.config.width]
        
        # Combine noise layers
        self.elevation = (noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2)
        
        # Apply Gaussian smoothing for realistic terrain
        self.elevation = gaussian_filter(self.elevation, sigma=2.0)
        
        # Normalize to 0-1 range
        self.elevation = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())
    
    def _create_mountain_ranges(self):
        """Create realistic mountain ranges"""
        for i in range(self.config.num_mountain_ranges):
            # Random starting point and direction
            start_x = random.randint(20, self.config.width - 20)
            start_y = random.randint(20, self.config.height - 20)
            
            # Random direction
            angle = random.uniform(0, 2 * math.pi)
            
            # Create mountain range
            for step in range(self.config.mountain_range_length):
                center_x = int(start_x + step * math.cos(angle))
                center_y = int(start_y + step * math.sin(angle))
                
                if 0 <= center_x < self.config.width and 0 <= center_y < self.config.height:
                    # Add mountain peaks with variation
                    for dx in range(-self.config.mountain_width, self.config.mountain_width):
                        for dy in range(-self.config.mountain_width, self.config.mountain_width):
                            x, y = center_x + dx, center_y + dy
                            if 0 <= x < self.config.width and 0 <= y < self.config.height:
                                distance = math.sqrt(dx*dx + dy*dy)
                                if distance < self.config.mountain_width:
                                    # Height decreases with distance from center
                                    height_factor = 1.0 - (distance / self.config.mountain_width)
                                    peak_height = 0.7 + 0.3 * random.random()
                                    self.elevation[y, x] = max(self.elevation[y, x], 
                                                             peak_height * height_factor)
    
    def _generate_base_terrain(self):
        """Generate base terrain types based on elevation and noise"""
        for y in range(self.config.height):
            for x in range(self.config.width):
                elevation = self.elevation[y, x]
                
                # Determine terrain based on elevation and location
                if elevation > 0.8:
                    self.terrain[y, x] = SNOW_MOUNTAIN
                elif elevation > 0.6:
                    self.terrain[y, x] = MOUNTAIN
                elif elevation > 0.45:
                    self.terrain[y, x] = FOOTHILLS
                elif elevation > 0.4:
                    # Check for alpine meadows near mountains
                    if self._near_mountains(x, y):
                        self.terrain[y, x] = ALPINE_MEADOW
                    else:
                        self.terrain[y, x] = FOREST
                else:
                    # Lower elevations - use noise for variety
                    noise_val = random.random()
                    
                    # Desert areas (inspired by New Austin)
                    if x < self.config.width * 0.3 and y > self.config.height * 0.6:
                        if noise_val < 0.8:
                            self.terrain[y, x] = DESERT
                        else:
                            self.terrain[y, x] = PLAINS
                    
                    # Swamp areas (inspired by Bayou Nwa)
                    elif x > self.config.width * 0.7 and y < self.config.height * 0.4:
                        if noise_val < 0.6:
                            self.terrain[y, x] = SWAMP
                        else:
                            self.terrain[y, x] = FOREST
                    
                    # Forest areas
                    elif noise_val < self.config.forest_coverage:
                        self.terrain[y, x] = FOREST
                    
                    # Default to plains
                    else:
                        self.terrain[y, x] = PLAINS
    
    def _near_mountains(self, x: int, y: int, radius: int = 5) -> bool:
        """Check if location is near mountains"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.config.width and 0 <= ny < self.config.height and
                    self.elevation[ny, nx] > 0.6):
                    return True
        return False
    
    def _create_water_systems(self):
        """Create rivers flowing from mountains to edges"""
        # Find mountain peaks as river sources
        mountain_peaks = []
        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.elevation[y, x] > 0.7:
                    mountain_peaks.append((x, y))
        
        # Create major rivers
        for _ in range(self.config.num_major_rivers):
            if mountain_peaks:
                start_x, start_y = random.choice(mountain_peaks)
                self._create_river(start_x, start_y, major=True)
        
        # Create minor rivers
        for _ in range(self.config.num_minor_rivers):
            if mountain_peaks:
                start_x, start_y = random.choice(mountain_peaks)
                self._create_river(start_x, start_y, major=False)
        
        # Add coastal water
        self._add_coastal_water()
    
    def _create_river(self, start_x: int, start_y: int, major: bool = True):
        """Create a river flowing from mountains to sea/edge"""
        current_x, current_y = start_x, start_y
        river_path = []
        width = self.config.river_width if major else 1
        
        # Flow towards lower elevation or edge
        for _ in range(200):  # Max river length
            river_path.append((current_x, current_y))
            
            # Find next position (flow downhill)
            best_x, best_y = current_x, current_y
            best_elevation = self.elevation[current_y, current_x]
            
            # Check 8 directions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x, new_y = current_x + dx, current_y + dy
                    
                    # Check bounds
                    if 0 <= new_x < self.config.width and 0 <= new_y < self.config.height:
                        elevation = self.elevation[new_y, new_x]
                        
                        # Prefer lower elevation with some randomness
                        if elevation < best_elevation - 0.01 or (elevation < best_elevation + 0.01 and random.random() < 0.3):
                            best_x, best_y = new_x, new_y
                            best_elevation = elevation
            
            current_x, current_y = best_x, best_y
            
            # Stop if we reach the edge or very low elevation
            if (current_x <= 1 or current_x >= self.config.width - 2 or 
                current_y <= 1 or current_y >= self.config.height - 2 or
                self.elevation[current_y, current_x] < 0.1):
                break
        
        # Place river tiles
        for x, y in river_path:
            for dx in range(-width//2, width//2 + 1):
                for dy in range(-width//2, width//2 + 1):
                    rx, ry = x + dx, y + dy
                    if 0 <= rx < self.config.width and 0 <= ry < self.config.height:
                        self.terrain[ry, rx] = RIVER
                        self.rivers.add((rx, ry))
    
    def _add_coastal_water(self):
        """Add water along map edges"""
        # Add water to edges
        for i in range(self.config.width):
            # Top and bottom edges
            if random.random() < 0.3:
                self.terrain[0, i] = WATER
                self.terrain[self.config.height - 1, i] = WATER
        
        for i in range(self.config.height):
            # Left and right edges
            if random.random() < 0.3:
                self.terrain[i, 0] = WATER
                self.terrain[i, self.config.width - 1] = WATER
    
    def _place_cities(self):
        """Place 7 cities strategically based on terrain and resources"""
        placed_cities = 0
        attempts = 0
        max_attempts = 100
        
        for city_template in self.city_templates:
            if placed_cities >= self.config.num_cities:
                break
                
            attempts = 0
            while attempts < max_attempts:
                x = random.randint(10, self.config.width - 10)
                y = random.randint(10, self.config.height - 10)
                
                # Check if location is suitable
                if self._is_suitable_city_location(x, y, city_template):
                    city = City(
                        name=city_template["name"],
                        x=x,
                        y=y,
                        population=city_template["population"],
                        city_type=city_template["type"],
                        terrain_preference=city_template["terrain"]
                    )
                    
                    self.cities.append(city)
                    
                    # Place city on map
                    size = 3 if city.population > 2000 else 2
                    for dx in range(-size, size + 1):
                        for dy in range(-size, size + 1):
                            cx, cy = x + dx, y + dy
                            if 0 <= cx < self.config.width and 0 <= cy < self.config.height:
                                if city.population > 3000:
                                    self.terrain[cy, cx] = CITY
                                elif city.population > 1000:
                                    self.terrain[cy, cx] = TOWN
                                else:
                                    self.terrain[cy, cx] = SETTLEMENT
                    
                    placed_cities += 1
                    print(f"Placed city: {city.name} at ({x}, {y})")
                    break
                
                attempts += 1
        
        print(f"Successfully placed {placed_cities} cities")
    
    def _is_suitable_city_location(self, x: int, y: int, city_template: Dict) -> bool:
        """Check if location is suitable for a city"""
        # Check minimum distance from other cities
        for city in self.cities:
            if euclidean((x, y), (city.x, city.y)) < 30:
                return False
        
        # Check terrain suitability
        terrain = self.terrain[y, x]
        elevation = self.elevation[y, x]
        
        # Cities shouldn't be in water or mountains
        if terrain in [WATER, MOUNTAIN, SNOW_MOUNTAIN]:
            return False
        
        # Check elevation constraints
        if elevation > 0.7 or elevation < 0.05:
            return False
        
        # Prefer locations near water for certain city types
        if city_template["type"] in ["industrial_port", "trading_post"]:
            near_water = False
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.config.width and 0 <= ny < self.config.height and
                        self.terrain[ny, nx] in [WATER, RIVER]):
                        near_water = True
                        break
                if near_water:
                    break
            if not near_water:
                return False
        
        return True
    
    def _create_road_network(self):
        """Create roads connecting cities and settlements"""
        print("Creating road network...")
        
        # Connect all cities with roads
        for i, city1 in enumerate(self.cities):
            for j, city2 in enumerate(self.cities[i+1:], i+1):
                if euclidean((city1.x, city1.y), (city2.x, city2.y)) < 100:
                    self._build_road(city1.x, city1.y, city2.x, city2.y)
        
        # Add additional roads for realism
        for city in self.cities:
            # Roads to map edges (trade routes)
            edge_targets = [
                (0, city.y),  # West edge
                (self.config.width - 1, city.y),  # East edge
                (city.x, 0),  # North edge
                (city.x, self.config.height - 1)  # South edge
            ]
            
            target = random.choice(edge_targets)
            if random.random() < 0.4:  # 40% chance for edge connection
                self._build_road(city.x, city.y, target[0], target[1])
    
    def _build_road(self, x1: int, y1: int, x2: int, y2: int):
        """Build a road between two points using A* pathfinding"""
        path = self._find_path(x1, y1, x2, y2)
        
        for x, y in path:
            if self.terrain[y, x] not in [WATER, RIVER, CITY, TOWN, SETTLEMENT]:
                self.terrain[y, x] = ROAD
                self.roads.add((x, y))
    
    def _create_railway_system(self):
        """Create railway lines connecting major cities"""
        print("Creating railway system...")
        
        # Sort cities by population
        major_cities = sorted(self.cities, key=lambda c: c.population, reverse=True)
        
        # Connect top cities with railways
        for i in range(min(len(major_cities), self.config.railway_lines + 1)):
            if i + 1 < len(major_cities):
                city1 = major_cities[i]
                city2 = major_cities[i + 1]
                self._build_railway(city1.x, city1.y, city2.x, city2.y)
        
        # Add one transcontinental railway
        if len(major_cities) >= 2:
            westmost = min(major_cities, key=lambda c: c.x)
            eastmost = max(major_cities, key=lambda c: c.x)
            self._build_railway(westmost.x, westmost.y, eastmost.x, eastmost.y)
    
    def _build_railway(self, x1: int, y1: int, x2: int, y2: int):
        """Build a railway between two points"""
        path = self._find_path(x1, y1, x2, y2, prefer_flat=True)
        
        for x, y in path:
            if self.terrain[y, x] not in [WATER, CITY, TOWN, SETTLEMENT]:
                self.terrain[y, x] = RAILWAY
                self.railways.add((x, y))
    
    def _find_path(self, x1: int, y1: int, x2: int, y2: int, prefer_flat: bool = False) -> List[Tuple[int, int]]:
        """Simple pathfinding for roads and railways"""
        path = []
        current_x, current_y = x1, y1
        
        while current_x != x2 or current_y != y2:
            path.append((current_x, current_y))
            
            # Move towards target
            dx = 1 if x2 > current_x else -1 if x2 < current_x else 0
            dy = 1 if y2 > current_y else -1 if y2 < current_y else 0
            
            # Add some randomness for natural-looking paths
            if random.random() < 0.1:
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            
            # Prefer flatter terrain for railways
            if prefer_flat:
                best_cost = float('inf')
                best_move = (dx, dy)
                
                for test_dx, test_dy in [(dx, dy), (0, dy), (dx, 0)]:
                    new_x = current_x + test_dx
                    new_y = current_y + test_dy
                    
                    if (0 <= new_x < self.config.width and 0 <= new_y < self.config.height):
                        elevation_change = abs(self.elevation[new_y, new_x] - self.elevation[current_y, current_x])
                        terrain_cost = TERRAIN_COSTS.get(self.terrain[new_y, new_x], 1.0)
                        total_cost = terrain_cost + elevation_change * 10
                        
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_move = (test_dx, test_dy)
                
                dx, dy = best_move
            
            current_x = max(0, min(self.config.width - 1, current_x + dx))
            current_y = max(0, min(self.config.height - 1, current_y + dy))
            
            # Prevent infinite loops
            if len(path) > 500:
                break
        
        path.append((current_x, current_y))
        return path
    
    def _add_terrain_details(self):
        """Add final terrain details and cleanup"""
        # Smooth transitions between terrain types
        for y in range(1, self.config.height - 1):
            for x in range(1, self.config.width - 1):
                if random.random() < 0.05:  # 5% chance for terrain variation
                    neighbors = [
                        self.terrain[y-1, x], self.terrain[y+1, x],
                        self.terrain[y, x-1], self.terrain[y, x+1]
                    ]
                    
                    # Find most common neighbor terrain
                    terrain_counts = {}
                    for terrain in neighbors:
                        terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
                    
                    most_common = max(terrain_counts, key=terrain_counts.get)
                    
                    # Sometimes adopt neighbor terrain for smoother transitions
                    if (terrain_counts[most_common] >= 3 and 
                        self.terrain[y, x] not in [WATER, RIVER, ROAD, RAILWAY, CITY, TOWN, SETTLEMENT]):
                        self.terrain[y, x] = most_common
    
    def get_cities(self) -> List[City]:
        """Return list of cities"""
        return self.cities
    
    def get_terrain_cost(self, x: int, y: int) -> float:
        """Get movement cost for terrain at position"""
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            return TERRAIN_COSTS.get(self.terrain[y, x], 1.0)
        return float('inf')
    
    def is_water(self, x: int, y: int) -> bool:
        """Check if position is water"""
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            return self.terrain[y, x] in [WATER, RIVER]
        return False
    
    def is_settlement(self, x: int, y: int) -> bool:
        """Check if position is a settlement"""
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            return self.terrain[y, x] in [CITY, TOWN, SETTLEMENT]
        return False

    def get_terrain_color(self, terrain_type: int) -> Tuple[int, int, int]:
        """Get the color for a given terrain type"""
        return TERRAIN_COLORS.get(terrain_type, (128, 128, 128))

def generate_rdr2_world(config: RDR2WorldConfig = None) -> Tuple[np.ndarray, RDR2WorldGenerator]:
    """Generate a complete RDR2-inspired world"""
    if config is None:
        config = RDR2WorldConfig()
    
    generator = RDR2WorldGenerator(config)
    terrain = generator.generate_world()
    
    return terrain, generator

if __name__ == "__main__":
    # Test generation
    config = RDR2WorldConfig()
    terrain, generator = generate_rdr2_world(config)
    
    print(f"Generated world: {config.width}x{config.height}")
    print(f"Cities: {len(generator.cities)}")
    print(f"Roads: {len(generator.roads)} tiles")
    print(f"Railways: {len(generator.railways)} tiles")
    print(f"Rivers: {len(generator.rivers)} tiles")
    
    # Print city information
    for city in generator.cities:
        print(f"  {city.name}: {city.city_type}, population {city.population} at ({city.x}, {city.y})") 
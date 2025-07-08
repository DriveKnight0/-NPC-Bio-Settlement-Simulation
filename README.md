# NPC - Bio-Inspired Settlement Simulation (Because Slime Mold Knows Best)

Welcome to the project where we let a blob of slime mold (yes, really) design your dream city's transport network. Why trust urban planners when you can trust Physarum Polycephalum, the Einstein of puddles?

## 🌟 Features (Because You Clearly Need Them)

### Core Simulation Systems
- **Physarum Intelligence**: Because if a slime can do it, why not your computer?
- **Multi-Agent System**: 8 agent roles, because 7 wasn't enough and 9 was too many.
- **Dynamic Trail Formation**: Watch as colored lines magically appear and pretend it's science.
- **GPU Acceleration**: For when you want your slime mold to think faster than you do.
- **Interactive Visualization**: Pygame window so you can stare at dots moving for hours.

### Agent Roles & Behaviors
| Role      | Population % | Trail Strength | Special Behavior           |
|-----------|--------------|---------------|---------------------------|
| Explorer  | 15%          | 0.8x          | Gets lost, finds stuff     |
| Trader    | 10%          | 2.0x          | Loves highways            |
| Settler   | 22%          | 1.0x          | Builds imaginary houses    |
| Miner     | 12%          | 1.0x          | Digs for invisible gold    |
| Farmer    | 23%          | 1.0x          | Grows crops, maybe         |
| Hunter    | 11%          | 1.0x          | Chases pixels             |
| Outlaw    | 5%           | 0.3x          | Avoids everyone           |
| Lawman    | 2%           | 1.5x          | Pretends to keep order     |

### City System (Prepare to Be Amazed)
Behold, the most creative city names ever:
- **City A**: Industrial Port (8,000 pop) - Swamp, because why not
- **City B**: Livestock Town (1,200 pop) - Plains, moo
- **City C**: Logging Town (800 pop) - Forest, trees beware
- **City D**: Desert Town (600 pop) - Desert, sand everywhere
- **City E**: Mining Town (1,500 pop) - Mountain, diggy diggy hole
- **City F**: Agricultural Town (1,000 pop) - Plains, more moo
- **City G**: Frontier City (3,000 pop) - Plains, wild wild west

## 🚀 Quick Start (Because Reading Is Hard)

### Prerequisites
```bash
pip install -r requirements.txt
```

**Required Dependencies:**
- `pygame` - For all your dot-watching needs
- `numpy` - Math, but faster
- `scipy` - Science, but lazier
- `cupy` - GPU, for when your CPU is tired

### Running the Simulation
```bash
python rdr2_main.py        # Main event
python cpu_main.py         # For potato PCs
python gpu_main.py         # For show-offs with GPUs
```

## 🎮 Controls (Because You're in Control, Allegedly)

| Key      | Function                        |
|----------|---------------------------------|
| SPACE    | Pause/unpause (take a break)    |
| R        | Toggle trails (hide the mess)   |
| 1-8      | Filter by role (pick your fav)  |
| +/-      | Trail intensity (make it pop)   |
| ESC      | Exit (rage quit)                |

## 🧠 Physarum Intelligence System (Slime Mold: 1, Humans: 0)

### Trail Evolution Phases
1. **Exploration** (Blue): Dots wander aimlessly
2. **Convergence** (Green): Dots pretend to know where they're going
3. **Stabilization** (Yellow): Dots agree on something, finally
4. **Highway Formation** (Red): Dots invent highways, take that, civil engineers

### Trail Mechanics
- **Deposition**: Agents leave trails, like snails but smarter
- **Decay**: Trails fade, just like your motivation
- **Diffusion**: Trails spread, because why stay put?
- **Reinforcement**: Popular paths get stronger, democracy in action

## 📊 Performance (Or Lack Thereof)

### System Requirements
- **Minimum**: 4GB RAM, hope, and patience
- **Recommended**: 8GB RAM, a GPU, and snacks
- **Optimal**: 16GB RAM, RTX GPU, and zero social life

### Simulation Scale
- **Agents**: Up to 1,000, because more is always better
- **World Size**: 300x300 grid, pixel paradise
- **Real-time**: 60 FPS, if you're lucky
- **Runtime**: 15-20 minutes to reach slime enlightenment

## 🔧 Configuration (For the Brave)

### Agent Population
Edit in `rdr2_main.py`:
```python
agent_roles = {
    'explorer': 0.15,   # 15% lost
    'trader': 0.10,     # 10% greedy
    'settler': 0.22,    # 22% homebodies
    # ... tweak as you wish
}
```

### Physarum Parameters
Mess with these in `PhysarumTrailSystem`:
```python
self.decay_rate = 0.995      # How long trails last
self.diffusion_rate = 0.1    # How much they spread
self.deposition_strength = 1.0  # How much agents care
```

## 📁 Project Structure (So You Can Pretend to Be Organized)

```
NPC/
├── rdr2_main.py              # Where the magic starts
├── rdr2_agent.py             # Agents pretending to be smart
├── rdr2_world_generator.py   # Cities, but generic
├── rdr2_gpu_simulation.py    # For GPU flexing
├── rdr2_visualizer.py        # Pretty colors
├── cpu_main.py               # For the rest of us
├── gpu_main.py               # For the 1%
├── simulation.py             # The real brains
├── agent.py                  # Agent basics
├── world_generator.py        # More world stuff
├── visualize.py              # More pretty colors
└── requirements.txt          # Install these or else
```

## 🔬 Scientific Background (Because This Is Totally Science)
- **Physarum Polycephalum**: Slime mold, the unsung hero
- **Swarm Intelligence**: Dots acting like they know what's up
- **Complex Adaptive Systems**: Fancy words for "it changes"
- **Spatial Economics**: Why cities exist, apparently

## 🎯 Use Cases (Impress Your Friends)
- **Urban Planning**: Or just confuse planners
- **Algorithm Research**: Because ants were too mainstream
- **Education**: Show students what chaos looks like
- **Game Development**: Procedural worlds, but slimier

## 🛠️ Troubleshooting (You'll Need This)
- **GPU not detected**: Did you try turning it off and on again?
- **Slow performance**: Blame your hardware, not the code
- **Import errors**: Install the requirements, please

### Performance Optimization
- Use GPU for 500+ agents, or just for bragging rights
- Lower trail resolution if your PC starts crying
- Shrink the world if you value your time

## 📝 License
This project is for educational and research purposes. If you make money, buy the slime mold a coffee.

## 🤝 Contributing
Fork it, break it, improve it, repeat. Slime mold would approve.

---

**Developed with bio-inspired intelligence and a healthy dose of sarcasm. 🧬🤖** 
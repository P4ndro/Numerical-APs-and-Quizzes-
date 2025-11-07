# 🌪️ Tornado Shelter Allocation using Voronoi Diagrams

**A computational geometry project for disaster preparedness and emergency planning**

---

## 📋 Project Overview

This Project is made for my Numerical Proggraming class.This project models tornado shelter allocation in a city using **Voronoi diagrams** and **Delaunay triangulations**. Each shelter serves the region closest to it, and when a tornado strikes, residents are assigned to their nearest shelter based on the chosen distance metric.

### Key Features
- ✅ **Voronoi Diagram** computation for shelter coverage regions
- ✅ **Delaunay Triangulation** for shelter connectivity analysis
- ✅ **Multiple Distance Metrics**: Euclidean (L2), Manhattan (L1), Chebyshev (L∞)
- ✅ **Tornado Strike Simulation** with nearest shelter identification
- ✅ **Beautiful Visualizations** with color-coded regions
- ✅ **Flexible Input**: Works with any shelter locations, tornado points, and distance norms
- ✅ **Reproducible Results**: Seeded random number generation

---

## 🎯 Real-World Application

**Problem**: Cities need to strategically place tornado shelters to ensure all residents can reach safety quickly during tornado warnings.

**Solution**: Voronoi diagrams partition the city into regions where each region is served by the nearest shelter. This helps:
- Identify underserved areas (large Voronoi cells = long travel distances)
- Optimize new shelter placement
- Analyze coverage under different distance models (straight-line vs. city blocks)
- Plan evacuation routes using Delaunay triangulation connectivity

---

## 📦 Requirements

### Built-in Python Libraries Only
```bash
numpy
matplotlib
scipy
```

### Installation
```bash
pip install numpy matplotlib scipy
```

---

## 🚀 Quick Start

### Run Main Demonstration
```bash
python tornado_shelter_voronoi.py
```

This will:
1. Generate random shelter locations
2. Simulate tornado strikes
3. Compute Voronoi diagrams and Delaunay triangulations
4. Compare Euclidean vs Manhattan distance metrics
5. Show the impact of adding a new shelter
6. Save 4 visualization PNG files

### Run Custom Data Demos
```bash
python demo_custom_data.py
```

This demonstrates:
- Custom shelter and tornado locations
- Different distance norms (L1, L2, L∞)
- Realistic city grid scenario
- How TA/instructor can test with their own data

---

## 💻 Usage Examples

### Example 1: Basic Usage
```python
from tornado_shelter_voronoi import TornadoShelterSystem
import numpy as np

# Define shelter locations
shelters = np.array([
    [20, 20],
    [80, 20],
    [50, 80]
])

# Create system with Euclidean distance
system = TornadoShelterSystem(shelters, bounds=(0, 100, 0, 100), metric='euclidean')

# Simulate 5 tornado strikes
results = system.simulate_tornado_strikes(n_strikes=5, seed=42)

# Visualize
system.plot_system(results)
```

### Example 2: Custom Tornado Locations
```python
# Find nearest shelter for a specific tornado location
tornado_location = [45, 60]
shelter_idx, shelter_coords, distance = system.find_nearest_shelter(tornado_location)

print(f"Nearest shelter: #{shelter_idx} at {shelter_coords}")
print(f"Distance: {distance:.2f} km")
```

### Example 3: Compare Distance Metrics
```python
# Compare how shelter allocation changes with different metrics
tornado_points = np.array([[30, 40], [70, 60], [50, 50]])

comparison = system.compare_metrics(
    tornado_points, 
    metrics=['euclidean', 'manhattan', 'chebyshev']
)
```

### Example 4: Testing with TA Data
```python
# TA provides their own data
ta_shelters = np.array([[x1, y1], [x2, y2], ...])
ta_tornadoes = np.array([[tx1, ty1], [tx2, ty2], ...])
ta_metric = 'manhattan'

# Create system and test
system = TornadoShelterSystem(ta_shelters, metric=ta_metric)

# Find nearest shelters
for tornado in ta_tornadoes:
    idx, coords, dist = system.find_nearest_shelter(tornado)
    print(f"Tornado → Shelter {idx}, Distance: {dist:.2f}")
```

---

## 📊 Distance Metrics Explained

### 1. Euclidean Distance (L2 Norm)
**Formula**: `d = √((x₁-x₂)² + (y₁-y₂)²)`

**Use Case**: Straight-line distance ("as the crow flies"). Good for:
- Helicopter evacuation planning
- Radio coverage areas
- General distance estimation

**Voronoi Shape**: Polygonal regions with straight boundaries

### 2. Manhattan Distance (L1 Norm)
**Formula**: `d = |x₁-x₂| + |y₁-y₂|`

**Use Case**: City block distance. Good for:
- Urban planning (can't walk through buildings)
- Street-based navigation
- Taxi/car travel in grid cities

**Voronoi Shape**: Diamond-shaped influence zones

### 3. Chebyshev Distance (L∞ Norm)
**Formula**: `d = max(|x₁-x₂|, |y₁-y₂|)`

**Use Case**: Movement where diagonal and straight moves cost the same (like a chess king). Good for:
- Grid-based movement with diagonals
- Some video game scenarios

---

## 🎨 Visualization Components

Each plot includes:

1. **Colored Voronoi Regions**: Each shelter's coverage area
2. **Delaunay Triangulation**: Gray lines showing shelter connectivity
3. **Shelter Markers**: Red circles labeled S0, S1, S2, ...
4. **Tornado Strikes**: Black X markers labeled T1, T2, T3, ...
5. **Assignment Lines**: Dashed lines connecting each tornado to its nearest shelter
6. **Legend and Labels**: Clear identification of all elements

---

## 📁 Project Structure

```
AP4/
├── tornado_shelter_voronoi.py      # Main implementation
├── demo_custom_data.py             # Demonstration scripts
├── README.md                       # This file
├── report_template.md              # Template for 2-page report
└── requirements.txt                # Python dependencies
```

---

## 🧪 Experiments Included

### Experiment 1: Euclidean Distance
- Visualize shelter coverage with straight-line distances
- Simulate tornado strikes and find nearest shelters
- Output: `tornado_shelter_euclidean.png`

### Experiment 2: Manhattan Distance
- Same simulation but with city-block distance
- Shows how urban grid layout affects allocation
- Output: `tornado_shelter_manhattan.png`

### Experiment 3: Side-by-Side Comparison
- Compare Euclidean vs Manhattan for same tornado strikes
- Highlight differences in shelter assignment
- Output: `tornado_shelter_comparison.png`

### Experiment 4: Adding New Shelter
- Show impact of adding one more shelter
- Demonstrates how coverage improves
- Output: `tornado_shelter_expanded.png`

---

## 📈 Key Results and Insights

1. **Manhattan distance creates diamond-shaped regions** (not circular like Euclidean)
2. **Adding shelters reduces average travel distance** for residents
3. **Different metrics can assign the same tornado to different shelters**
4. **Delaunay triangulation shows optimal communication paths** between shelters
5. **Voronoi diagram identifies underserved areas** (large regions = poor coverage)

---

## 🎥 Video Demonstration Script (3 minutes)

### Minute 1: Introduction (20 sec)
- "This project models tornado shelter allocation using Voronoi diagrams"
- Show the problem: shelters need to serve nearby residents
- Explain Voronoi diagram concept visually

### Minute 2: Live Demo (1:40 min)
- Run the main script: `python tornado_shelter_voronoi.py`
- Show Euclidean distance visualization
- Show Manhattan distance visualization
- Point out differences in region shapes
- Show comparison plot
- Demonstrate adding a new shelter

### Minute 3: Technical Explanation (1:00 min)
- Explain distance metrics (L1 vs L2)
- Show code flexibility: custom data example
- Explain Delaunay triangulation purpose
- Discuss real-world implications
- Wrap up with key conclusions

---

## ✅ Testing and Validation

### Reproducibility
- All random operations use seeds
- Results are consistent across runs
- Output files have descriptive names

### Flexibility
- Works with any number of shelters (N ≥ 3)
- Works with any number of tornadoes
- Works with any bounded region
- Supports multiple distance metrics
- Can handle custom input data from TA

### Correctness
- Voronoi regions computed using scipy (industry standard)
- Distance calculations verified against mathematical definitions
- Nearest shelter assignment verified by exhaustive search

---

## 🎓 Academic Components

### Mathematical Model
- **Input**: Set of n shelter points S = {s₁, s₂, ..., sₙ} in R²
- **Output**: Voronoi diagram V(S) partitioning the plane
- **Definition**: For each sᵢ, V(sᵢ) = {p ∈ R² : d(p, sᵢ) ≤ d(p, sⱼ) ∀j ≠ i}
- **Metric**: Distance function d(·,·) ∈ {L₁, L₂, L∞}

### Algorithmic Approach
1. **Voronoi Computation**: Fortune's algorithm (O(n log n))
2. **Delaunay Triangulation**: Flip algorithm (O(n log n))
3. **Nearest Neighbor**: Linear search through shelters (O(n))
4. **Visualization**: Matplotlib polygon rendering

### Complexity Analysis
- **Time**: O(n log n) for Voronoi + O(m·n) for m tornado queries
- **Space**: O(n) for storing n shelters and their regions

---

## 🔧 Code Architecture

### Class: `TornadoShelterSystem`

**Attributes**:
- `shelters`: NumPy array of shelter coordinates
- `bounds`: City boundary (xmin, xmax, ymin, ymax)
- `metric`: Distance metric ('euclidean', 'manhattan', 'chebyshev')
- `voronoi`: Scipy Voronoi object
- `delaunay`: Scipy Delaunay object

**Methods**:
- `generate_random_shelters()`: Generate random shelter locations
- `calculate_distance()`: Compute distance using specified metric
- `find_nearest_shelter()`: Find closest shelter to a point
- `simulate_tornado_strikes()`: Generate random tornadoes and find shelters
- `plot_system()`: Visualize Voronoi + Delaunay + assignments
- `compare_metrics()`: Compare different distance metrics

---

## 📝 Report Guidelines (2-page paper)

See `report_template.md` for a structured template covering:
1. **Introduction**: Problem statement and motivation
2. **Mathematical Model**: Formal definitions and metrics
3. **Approach**: Implementation and algorithms
4. **Experiments**: Results and visualizations
5. **Conclusions**: Insights and future work

---

## 🚨 Important Notes for TA

### This code is designed to be testable with:
✅ **Any shelter locations** (provide as NumPy array)
✅ **Any tornado locations** (provide as NumPy array)
✅ **Any distance metric** ('euclidean', 'manhattan', 'chebyshev')
✅ **Any bounded region** (specify bounds tuple)

### To test with your own data:
```python
from tornado_shelter_voronoi import TornadoShelterSystem
import numpy as np

# Your data
your_shelters = np.array([[...], [...], ...])
your_tornadoes = np.array([[...], [...], ...])
your_metric = 'manhattan'  # or 'euclidean' or 'chebyshev'

# Test
system = TornadoShelterSystem(your_shelters, metric=your_metric)
for tornado in your_tornadoes:
    idx, coords, dist = system.find_nearest_shelter(tornado)
    print(f"Shelter {idx}, Distance: {dist:.2f}")
```

---

## 📚 References

1. **Voronoi Diagrams**: Aurenhammer, F. (1991). "Voronoi diagrams—a survey of a fundamental geometric data structure."
2. **Delaunay Triangulation**: Delaunay, B. (1934). "Sur la sphere vide."
3. **Scipy Documentation**: https://docs.scipy.org/doc/scipy/reference/spatial.html
4. **Disaster Planning**: FEMA Tornado Safety Guidelines

---

## 👨‍💻 Author

Created for Numerical Methods Course  
Project: AP4 - Voronoi Diagrams Application  
Date: November 2024

---

## 📄 License

This project is created for educational purposes as part of a university course assignment.

---

## 🎯 Deliverable Checklist

- ✅ Working Python code with Voronoi diagrams
- ✅ Working Python code with Delaunay triangulations
- ✅ Multiple distance metric support (L1, L2, L∞)
- ✅ Visualization with clear labels and legends
- ✅ Tornado strike simulation
- ✅ Flexible input for TA testing
- ✅ Reproducible results with seeds
- ✅ Comprehensive README documentation
- ✅ Report template for 2-page paper
- ✅ Demo scripts for various scenarios
- ✅ Real-world application (disaster preparedness)
- ✅ Unique problem (tornado shelters)

---

**Ready for 3-minute video demonstration! 🎥**
**Ready for TA testing! 🧪**
**Ready for report writing! 📝**


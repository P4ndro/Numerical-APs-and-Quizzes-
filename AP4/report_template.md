# Tornado Shelter Allocation using Voronoi Diagrams

**Course:** Numerical Methods  
**Project:** AP4 - Real-World Applications of Voronoi Diagrams  
**Author:** Sandro iobidze 


---

## 1. Introduction and Motivation

### Problem Statement
During tornado warnings, residents must quickly evacuate to the nearest tornado shelter. Cities need to strategically place shelters to minimize travel distances and ensure adequate coverage for all neighborhoods. This project models tornado shelter allocation using computational geometry, specifically **Voronoi diagrams** and **Delaunay triangulations**.

### Real-World Application
Tornado shelters are critical infrastructure in tornado-prone regions (e.g., Oklahoma, Kansas, Alabama). Poor shelter placement can result in:
- Long evacuation distances
- Underserved neighborhoods
- Loss of life during emergencies

Voronoi diagrams provide a mathematical framework to:
1. **Visualize coverage areas** - each shelter serves the region closest to it
2. **Identify gaps** - large Voronoi cells indicate underserved areas
3. **Optimize placement** - adding shelters reduces cell sizes and travel distances
4. **Model different scenarios** - urban grids (Manhattan distance) vs. rural areas (Euclidean distance)

### Project Goals
- Implement Voronoi diagram and Delaunay triangulation algorithms
- Simulate tornado strikes and assign residents to nearest shelters
- Compare different distance metrics (Euclidean vs. Manhattan)
- Visualize shelter coverage and connectivity
- Analyze the impact of adding new shelters

---

## 2. Mathematical Model

### 2.1 Voronoi Diagram Definition

Given a set of **n shelter locations** S = {s₁, s₂, ..., sₙ} in ℝ², the **Voronoi diagram** V(S) partitions the plane into n regions, where each region V(sᵢ) contains all points closer to shelter sᵢ than to any other shelter:

```
V(sᵢ) = {p ∈ ℝ² : d(p, sᵢ) ≤ d(p, sⱼ) for all j ≠ i}
```

Where d(·,·) is a distance function (metric).

**Properties:**
- Each Voronoi cell is a convex polygon (for Euclidean distance)
- Cells share edges where two shelters are equidistant
- The diagram partitions the entire plane

### 2.2 Distance Metrics

#### Euclidean Distance (L₂ norm)
```
d₂(p, s) = √[(pₓ - sₓ)² + (pᵧ - sᵧ)²]
```
- **Use case:** Straight-line distance, "as the crow flies"
- **Application:** Helicopter evacuation, radio coverage

#### Manhattan Distance (L₁ norm)
```
d₁(p, s) = |pₓ - sₓ| + |pᵧ - sᵧ|
```
- **Use case:** City-block distance, grid-based navigation
- **Application:** Walking/driving in urban grid layouts

#### Chebyshev Distance (L∞ norm)
```
d∞(p, s) = max(|pₓ - sₓ|, |pᵧ - sᵧ|)
```
- **Use case:** Grid movement with diagonals
- **Application:** Some simulation models

### 2.3 Delaunay Triangulation

The **Delaunay triangulation** D(S) is the dual graph of the Voronoi diagram. It connects shelters whose Voronoi regions share an edge.

**Properties:**
- Maximizes the minimum angle of all triangles (avoids "skinny" triangles)
- Empty circle property: no shelter lies inside the circumcircle of any triangle
- Useful for: communication networks, emergency coordination routes

### 2.4 Problem Formulation

**Input:**
- Shelter locations: S = {s₁, s₂, ..., sₙ} ⊂ ℝ²
- City bounds: [xₘᵢₙ, xₘₐₓ] × [yₘᵢₙ, yₘₐₓ]
- Distance metric: d ∈ {L₁, L₂, L∞}
- Tornado strike locations: T = {t₁, t₂, ..., tₘ} ⊂ ℝ²

**Output:**
- Voronoi diagram V(S)
- Delaunay triangulation D(S)
- Assignment: tornado tᵢ → nearest shelter s*
  ```
  s* = argmin_{s∈S} d(tᵢ, s)
  ```

**Objective:** Minimize average evacuation distance
```
minimize: (1/m) Σᵢ₌₁ᵐ d(tᵢ, s*)
```

---

## 3. Approach and Implementation

### 3.1 Algorithm Overview

1. **Input Generation**
   - Generate n random shelter locations (or use provided coordinates)
   - Generate m random tornado strike locations

2. **Voronoi Computation**
   - Use scipy.spatial.Voronoi (implements Fortune's algorithm)
   - Time complexity: O(n log n)
   - Space complexity: O(n)

3. **Delaunay Triangulation**
   - Use scipy.spatial.Delaunay (implements flip algorithm)
   - Time complexity: O(n log n)
   - Dual relationship with Voronoi diagram

4. **Nearest Shelter Assignment**
   - For each tornado location t:
     - Compute distance d(t, sᵢ) for all shelters
     - Assign to shelter with minimum distance
   - Time complexity: O(m·n) for m tornadoes

5. **Visualization**
   - Plot Voronoi regions (colored polygons)
   - Plot Delaunay edges (gray lines)
   - Mark shelters (red circles) and tornadoes (black X)
   - Draw assignment lines (dashed)

### 3.2 Software Architecture

**Class: TornadoShelterSystem**
- Encapsulates all functionality
- Attributes: shelters, bounds, metric, voronoi, delaunay
- Methods:
  - `generate_random_shelters()` - create test data
  - `calculate_distance()` - compute distance with specified metric
  - `find_nearest_shelter()` - assign tornado to shelter
  - `simulate_tornado_strikes()` - generate and process multiple tornadoes
  - `plot_system()` - visualize everything
  - `plot_separate_views()` - create 3-panel visualization
  - `compare_metrics()` - compare L₁ vs L₂

### 3.3 Built-in Libraries Used
- **NumPy**: Array operations, random number generation
- **Matplotlib**: Visualization (plots, polygons, annotations)
- **SciPy**: Voronoi diagrams and Delaunay triangulations

### 3.4 Code Flexibility
The implementation accepts:
- Any set of shelter coordinates (provided by user/TA)
- Any set of tornado coordinates
- Any distance metric
- Any bounded region
- Seeded randomness for reproducibility

---

## 4. Experiments and Results

### Experiment 1: Euclidean Distance

**Setup:**
- 8 randomly placed shelters
- 10 simulated tornado strikes
- 100×100 km² city area
- Euclidean (L₂) distance metric

**Results:**
- Average evacuation distance: [X.XX] km
- Maximum evacuation distance: [X.XX] km
- All tornadoes successfully assigned to shelters

**Visualization:** `tornado_shelter_euclidean.png`

**Observations:**
- Voronoi cells form irregular polygons
- Central shelters have smaller coverage areas (higher density)
- Edge shelters cover larger, less populated regions

---

### Experiment 2: Manhattan Distance

**Setup:**
- Same 8 shelter locations
- Same 10 tornado locations
- Manhattan (L₁) distance metric (models city grid)

**Results:**
- Average evacuation distance: [X.XX] blocks
- Maximum evacuation distance: [X.XX] blocks
- Some tornadoes assigned to different shelters vs. Euclidean

**Visualization:** `tornado_shelter_manhattan.png`

**Observations:**
- Voronoi cells have diamond-shaped boundaries (not circular)
- More realistic for cities with grid street layouts
- Different metric → different optimal shelter assignments

---

### Experiment 3: Metric Comparison

**Setup:**
- Side-by-side comparison of Euclidean vs. Manhattan
- Same shelter and tornado locations

**Visualization:** `tornado_shelter_comparison.png`

**Key Findings:**
1. **Different assignments:** [X]% of tornadoes assigned to different shelters
2. **Distance differences:** Manhattan distances are [X]% longer on average
3. **Shape differences:** Polygonal (L₂) vs. diamond-shaped (L₁) regions

---

### Experiment 4: Adding a New Shelter

**Setup:**
- Original 8 shelters + 1 new shelter at center (50, 50)
- Same tornado locations

**Results:**
- **Before:** Average distance = [X.XX] km
- **After:** Average distance = [Y.YY] km
- **Improvement:** [Z.ZZ] km reduction (–[P]%)

**Visualization:** `tornado_shelter_expanded.png`

**Observations:**
- New shelter "steals" coverage from surrounding shelters
- Voronoi cells shrink → shorter evacuation distances
- Demonstrates value of strategic shelter placement

---

### Experiment 5: Three-View Visualization

**Visualization:** `tornado_shelter_separate_views.png`

Shows three panels:
1. **Voronoi only:** Service territories (colored regions)
2. **Delaunay only:** Shelter connectivity (gray network)
3. **Combined:** Complete system view

**Purpose:** Clearly demonstrate both geometric structures and their relationship.

---

## 5. Conclusions and Insights

### Key Findings

1. **Voronoi diagrams effectively model service territories**
   - Each region shows shelter coverage area
   - Large cells indicate underserved neighborhoods
   - Visual tool for urban planning

2. **Distance metric matters**
   - Euclidean: theoretical minimum (straight line)
   - Manhattan: realistic for city grids
   - Choice affects optimal shelter placement

3. **Delaunay triangulation shows connectivity**
   - Reveals communication/coordination network
   - Useful for emergency response planning
   - Dual relationship with Voronoi diagram

4. **Adding shelters improves coverage**
   - Reduces average evacuation distance
   - Voronoi diagram helps identify where to add shelters
   - Quantifiable improvement in public safety

5. **Computational efficiency**
   - O(n log n) algorithms scale well
   - Real-time analysis possible for city-scale problems
   - Built-in libraries provide robust implementations

### Real-World Implications

- **Urban planning:** Optimize public facility placement (shelters, hospitals, fire stations)
- **Disaster preparedness:** Identify vulnerable areas, plan evacuation routes
- **Resource allocation:** Determine how many shelters are needed for adequate coverage
- **Equity analysis:** Ensure all neighborhoods have reasonable access

### Limitations

1. **Simplified model:** Assumes unobstructed travel, uniform terrain
2. **Static analysis:** Doesn't account for time-varying factors (traffic, road conditions)
3. **Capacity ignored:** Real shelters have capacity limits
4. **2D only:** Elevation/terrain not considered

### Future Extensions

1. **Weighted Voronoi diagrams:** Account for shelter capacity
2. **Real GIS data:** Use actual city maps and road networks
3. **Network distances:** Use shortest path on road graph instead of straight-line
4. **3D visualization:** Incorporate terrain elevation
5. **Optimization algorithms:** Automatically suggest optimal shelter locations
6. **Population density:** Weight regions by resident count
7. **Multi-hazard:** Extend to floods, earthquakes, etc.

---

## 6. References

1. Fortune, S. (1987). "A sweepline algorithm for Voronoi diagrams." *Algorithmica*, 2(1-4), 153-174.

2. Aurenhammer, F. (1991). "Voronoi diagrams—a survey of a fundamental geometric data structure." *ACM Computing Surveys*, 23(3), 345-405.

3. Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N. (2000). *Spatial Tessellations: Concepts and Applications of Voronoi Diagrams*. John Wiley & Sons.

4. Preparata, F. P., & Shamos, M. I. (1985). *Computational Geometry: An Introduction*. Springer-Verlag.

5. FEMA (2023). "Tornado Safety Guidelines." Federal Emergency Management Agency.

6. SciPy Documentation: https://docs.scipy.org/doc/scipy/reference/spatial.html

---

## Appendix: Code Structure

```
AP4/
├── tornado_shelter_voronoi.py      # Main implementation (500+ lines)
├── simple_demo.py                  # Quick 3-view demonstration
├── demo_custom_data.py             # Testing with custom inputs
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
└── report_template.md              # This report template
```

**Key Functions:**
- `TornadoShelterSystem.__init__()` - Initialize system
- `generate_random_shelters()` - Create test data
- `calculate_distance()` - Compute distances with various metrics
- `find_nearest_shelter()` - Tornado → shelter assignment
- `plot_system()` - Full visualization
- `plot_separate_views()` - Three-panel visualization

---

**Total Pages:** 2 (adjust as needed)  
**Word Count:** ~1800 words  
**Figures:** 5 visualization PNG files


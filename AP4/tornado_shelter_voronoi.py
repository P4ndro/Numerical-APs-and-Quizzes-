"""
Tornado Shelter Allocation using Voronoi Diagrams and Delaunay Triangulations

Model how tornado shelters serve nearby regions using Voronoi diagrams.
Each shelter covers the area closest to it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')


class TornadoShelterSystem:
    """
    A system for modeling tornado shelter allocation using Voronoi diagrams.
    
    Attributes:
        shelters (np.ndarray): Array of shelter coordinates (N x 2)
        bounds (tuple): Boundary of the city area (xmin, xmax, ymin, ymax)
        voronoi (Voronoi): Scipy Voronoi object
        delaunay (Delaunay): Scipy Delaunay object
        metric (str): Distance metric to use ('euclidean' or 'manhattan')
    """
    
    def __init__(self, shelters=None, bounds=(0, 100, 0, 100), metric='euclidean'):
        """
        Initialize the tornado shelter system.
        
        Parameters:
            shelters (np.ndarray): Array of shelter coordinates, or None to generate random
            bounds (tuple): (xmin, xmax, ymin, ymax) for the city area
            metric (str): 'euclidean' (L2) or 'manhattan' (L1) distance
        """
        self.bounds = bounds
        self.metric = metric.lower()
        
        if shelters is None:
            raise ValueError("Shelters must be provided")
        
        self.shelters = np.array(shelters)
        if self.shelters.ndim == 1:
            self.shelters = self.shelters.reshape(-1, 2)
        
        # Compute Voronoi diagram and Delaunay triangulation
        self.voronoi = Voronoi(self.shelters)
        self.delaunay = Delaunay(self.shelters)
        
    @staticmethod
    def generate_random_shelters(n_shelters, bounds=(0, 100, 0, 100), seed=None):
        """
        Generate random shelter locations within bounds.
        
        Parameters:
            n_shelters (int): Number of shelters to generate
            bounds (tuple): (xmin, xmax, ymin, ymax)
            seed (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: Array of shelter coordinates (n_shelters x 2)
        """
        if seed is not None:
            np.random.seed(seed)
        
        xmin, xmax, ymin, ymax = bounds
        x = np.random.uniform(xmin, xmax, n_shelters)
        y = np.random.uniform(ymin, ymax, n_shelters)
        
        return np.column_stack([x, y])
    
    def calculate_distance(self, point1, point2):
        """
        Calculate distance between two points using the specified metric.
        
        Parameters:
            point1 (np.ndarray): First point (x, y)
            point2 (np.ndarray): Second point (x, y)
            
        Returns:
            float: Distance between points
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        if self.metric == 'euclidean':
            # L2 norm: sqrt((x1-x2)^2 + (y1-y2)^2)
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.metric == 'manhattan':
            # L1 norm: |x1-x2| + |y1-y2|
            return np.sum(np.abs(point1 - point2))
        elif self.metric == 'chebyshev':
            # L-infinity norm: max(|x1-x2|, |y1-y2|)
            return np.max(np.abs(point1 - point2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def find_nearest_shelter(self, tornado_point):
        """
        Find the nearest shelter to a tornado strike point.
        
        Parameters:
            tornado_point (tuple or np.ndarray): (x, y) coordinates of tornado
            
        Returns:
            tuple: (shelter_index, shelter_coordinates, distance)
        """
        tornado_point = np.array(tornado_point)
        
        # Calculate distances to all shelters
        distances = [self.calculate_distance(tornado_point, shelter) 
                    for shelter in self.shelters]
        
        # Find the nearest shelter
        nearest_idx = np.argmin(distances)
        nearest_shelter = self.shelters[nearest_idx]
        nearest_distance = distances[nearest_idx]
        
        return nearest_idx, nearest_shelter, nearest_distance
    
    def simulate_tornado_strikes(self, n_strikes, seed=None):
        """
        Simulate random tornado strikes and find nearest shelters.
        
        Parameters:
            n_strikes (int): Number of tornado strikes to simulate
            seed (int): Random seed for reproducibility
            
        Returns:
            list: List of tuples (tornado_point, shelter_idx, shelter_coords, distance)
        """
        if seed is not None:
            np.random.seed(seed)
        
        xmin, xmax, ymin, ymax = self.bounds
        
        results = []
        for i in range(n_strikes):
            # Generate random tornado location
            tornado_x = np.random.uniform(xmin, xmax)
            tornado_y = np.random.uniform(ymin, ymax)
            tornado_point = np.array([tornado_x, tornado_y])
            
            # Find nearest shelter
            shelter_idx, shelter_coords, distance = self.find_nearest_shelter(tornado_point)
            
            results.append({
                'tornado_point': tornado_point,
                'shelter_idx': shelter_idx,
                'shelter_coords': shelter_coords,
                'distance': distance
            })
            
            print(f"Tornado #{i+1} at ({tornado_x:.2f}, {tornado_y:.2f}) "
                  f"-> Shelter #{shelter_idx} at ({shelter_coords[0]:.2f}, {shelter_coords[1]:.2f}) "
                  f"[Distance: {distance:.2f} using {self.metric}]")
        
        return results
    
    def plot_system(self, tornado_results=None, title=None, figsize=(12, 10)):
        """
        Visualize the shelter system with Voronoi diagram and Delaunay triangulation.
        
        Parameters:
            tornado_results (list): Results from simulate_tornado_strikes()
            title (str): Custom title for the plot
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set bounds
        xmin, xmax, ymin, ymax = self.bounds
        ax.set_xlim(xmin - 5, xmax + 5)
        ax.set_ylim(ymin - 5, ymax + 5)
        
        # Plot Voronoi diagram
        self._plot_voronoi_regions(ax)
        
        # Plot Delaunay triangulation
        self._plot_delaunay_triangulation(ax)
        
        # Plot shelters
        ax.scatter(self.shelters[:, 0], self.shelters[:, 1], 
                  c='red', s=200, marker='o', edgecolors='darkred', 
                  linewidths=2, label='Shelters', zorder=5)
        
        # Label shelters
        for i, shelter in enumerate(self.shelters):
            ax.annotate(f'S{i}', (shelter[0], shelter[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='darkred')
        
        # Plot tornado strikes if provided
        if tornado_results:
            tornado_points = np.array([r['tornado_point'] for r in tornado_results])
            ax.scatter(tornado_points[:, 0], tornado_points[:, 1],
                      c='black', s=150, marker='x', linewidths=3,
                      label='Tornado Strikes', zorder=6)
            
            # Draw lines from tornado to nearest shelter
            for i, result in enumerate(tornado_results):
                tornado = result['tornado_point']
                shelter = result['shelter_coords']
                ax.plot([tornado[0], shelter[0]], [tornado[1], shelter[1]],
                       'k--', alpha=0.4, linewidth=1, zorder=4)
                
                # Label tornado
                ax.annotate(f'T{i+1}', (tornado[0], tornado[1]),
                           xytext=(5, -15), textcoords='offset points',
                           fontsize=8, color='black')
        
        # Set title and labels
        if title is None:
            title = f'Tornado Shelter Allocation - {self.metric.capitalize()} Distance'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate (km)', fontsize=12)
        ax.set_ylabel('Y Coordinate (km)', fontsize=12)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def _voronoi_finite_polygons_2d(self, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
        Clips infinite regions to a bounding box.
        
        Returns:
            regions (list): Indices of vertices in each revised region
            vertices (array): Coordinates for revised vertices
        """
        if radius is None:
            xmin, xmax, ymin, ymax = self.bounds
            radius = max(xmax - xmin, ymax - ymin) * 2
        
        new_regions = []
        new_vertices = self.voronoi.vertices.tolist()
        
        center = self.voronoi.points.mean(axis=0)
        
        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))
        
        # Reconstruct infinite regions
        for p1, region in enumerate(self.voronoi.point_region):
            vertices = self.voronoi.regions[region]
            
            if all(v >= 0 for v in vertices):
                # Finite region
                new_regions.append(vertices)
                continue
            
            # Reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # Finite ridge: already in the region
                    continue
                
                # Compute the missing endpoint of an infinite ridge
                t = self.voronoi.points[p2] - self.voronoi.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                
                midpoint = self.voronoi.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.voronoi.vertices[v2] + direction * radius
                
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            
            # Sort region vertices by angle around the point
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            
            new_regions.append(new_region.tolist())
        
        return new_regions, np.asarray(new_vertices)
    
    def _plot_voronoi_regions(self, ax):
        """Plot Voronoi regions with different colors, properly handling infinite regions."""
        # Generate colors for each shelter
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.shelters)))
        
        # Get finite polygons
        regions, vertices = self._voronoi_finite_polygons_2d()
        
        # Color each region
        for region_idx, region in enumerate(regions):
            if len(region) >= 3:
                polygon_vertices = vertices[region]
                polygon = Polygon(polygon_vertices, facecolor=colors[region_idx],
                                edgecolor='black', alpha=0.4, linewidth=1.5)
                ax.add_patch(polygon)
    
    def _plot_delaunay_triangulation(self, ax):
        """Plot Delaunay triangulation."""
        # Get triangulation edges
        edges = set()
        for simplex in self.delaunay.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                edges.add(edge)
        
        # Plot edges
        for edge in edges:
            points = self.shelters[list(edge)]
            ax.plot(points[:, 0], points[:, 1], 'gray', 
                   linewidth=1, alpha=0.6, zorder=3)
    
    def compare_metrics(self, tornado_points, metrics=['euclidean', 'manhattan']):
        """
        Compare shelter allocation under different distance metrics.
        
        Parameters:
            tornado_points (np.ndarray): Array of tornado coordinates
            metrics (list): List of metrics to compare
            
        Returns:
            dict: Results for each metric
        """
        results = {}
        
        for metric in metrics:
            print(f"\nAnalyzing with {metric.upper()} distance metric")
            
            # Create new system with this metric
            system = TornadoShelterSystem(self.shelters, self.bounds, metric)
            
            # Find nearest shelters for each tornado
            metric_results = []
            for i, tornado in enumerate(tornado_points):
                shelter_idx, shelter_coords, distance = system.find_nearest_shelter(tornado)
                metric_results.append({
                    'tornado_point': tornado,
                    'shelter_idx': shelter_idx,
                    'shelter_coords': shelter_coords,
                    'distance': distance
                })
                
                print(f"Tornado #{i+1} -> Shelter #{shelter_idx} "
                      f"[Distance: {distance:.2f}]")
            
            results[metric] = {
                'system': system,
                'results': metric_results
            }
        
        return results
    
    def plot_separate_views(self, tornado_results=None, figsize=(18, 6)):
        """
        Create three separate visualizations:
        1. Voronoi diagram only (territories)
        2. Delaunay triangulation only (connections)
        3. Combined view (both)
        
        Parameters:
            tornado_results (list): Results from simulate_tornado_strikes()
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure with 3 subplots
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        xmin, xmax, ymin, ymax = self.bounds
        
        # Subplot 1: Voronoi Diagram Only
        ax1 = axes[0]
        ax1.set_xlim(xmin - 5, xmax + 5)
        ax1.set_ylim(ymin - 5, ymax + 5)
        
        # Plot Voronoi regions
        self._plot_voronoi_regions(ax1)
        
        # Plot shelters
        ax1.scatter(self.shelters[:, 0], self.shelters[:, 1], 
                   c='red', s=200, marker='o', edgecolors='darkred', 
                   linewidths=2, label='Shelters', zorder=5)
        
        # Label shelters
        for i, shelter in enumerate(self.shelters):
            ax1.annotate(f'S{i}', (shelter[0], shelter[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='darkred')
        
        # Plot tornado strikes if provided
        if tornado_results:
            tornado_points = np.array([r['tornado_point'] for r in tornado_results])
            ax1.scatter(tornado_points[:, 0], tornado_points[:, 1],
                       c='black', s=150, marker='x', linewidths=3,
                       label='Tornado Strikes', zorder=6)
            
            for i, result in enumerate(tornado_results):
                tornado = result['tornado_point']
                ax1.annotate(f'T{i+1}', (tornado[0], tornado[1]),
                           xytext=(5, -15), textcoords='offset points',
                           fontsize=8, color='black')
        
        ax1.set_title('Voronoi Diagram\n(Service Territories)', 
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('X Coordinate (km)', fontsize=10)
        ax1.set_ylabel('Y Coordinate (km)', fontsize=10)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.set_aspect('equal')
        
        # Subplot 2: Delaunay Triangulation Only
        ax2 = axes[1]
        ax2.set_xlim(xmin - 5, xmax + 5)
        ax2.set_ylim(ymin - 5, ymax + 5)
        
        # Plot Delaunay triangulation
        self._plot_delaunay_triangulation(ax2)
        
        # Plot shelters
        ax2.scatter(self.shelters[:, 0], self.shelters[:, 1], 
                   c='red', s=200, marker='o', edgecolors='darkred', 
                   linewidths=2, label='Shelters', zorder=5)
        
        # Label shelters
        for i, shelter in enumerate(self.shelters):
            ax2.annotate(f'S{i}', (shelter[0], shelter[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='darkred')
        
        # Plot tornado strikes if provided
        if tornado_results:
            tornado_points = np.array([r['tornado_point'] for r in tornado_results])
            ax2.scatter(tornado_points[:, 0], tornado_points[:, 1],
                       c='black', s=150, marker='x', linewidths=3,
                       label='Tornado Strikes', zorder=6)
            
            for i, result in enumerate(tornado_results):
                tornado = result['tornado_point']
                ax2.annotate(f'T{i+1}', (tornado[0], tornado[1]),
                           xytext=(5, -15), textcoords='offset points',
                           fontsize=8, color='black')
        
        ax2.set_title('Delaunay Triangulation\n(Shelter Connections)', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('X Coordinate (km)', fontsize=10)
        ax2.set_ylabel('Y Coordinate (km)', fontsize=10)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.set_aspect('equal')
        
        # Subplot 3: Combined View
        ax3 = axes[2]
        ax3.set_xlim(xmin - 5, xmax + 5)
        ax3.set_ylim(ymin - 5, ymax + 5)
        
        # Plot Voronoi regions
        self._plot_voronoi_regions(ax3)
        
        # Plot Delaunay triangulation
        self._plot_delaunay_triangulation(ax3)
        
        # Plot shelters
        ax3.scatter(self.shelters[:, 0], self.shelters[:, 1], 
                   c='red', s=200, marker='o', edgecolors='darkred', 
                   linewidths=2, label='Shelters', zorder=5)
        
        # Label shelters
        for i, shelter in enumerate(self.shelters):
            ax3.annotate(f'S{i}', (shelter[0], shelter[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='darkred')
        
        # Plot tornado strikes if provided
        if tornado_results:
            tornado_points = np.array([r['tornado_point'] for r in tornado_results])
            ax3.scatter(tornado_points[:, 0], tornado_points[:, 1],
                       c='black', s=150, marker='x', linewidths=3,
                       label='Tornado Strikes', zorder=6)
            
            # Draw lines from tornado to nearest shelter
            for i, result in enumerate(tornado_results):
                tornado = result['tornado_point']
                shelter = result['shelter_coords']
                ax3.plot([tornado[0], shelter[0]], [tornado[1], shelter[1]],
                        'k--', alpha=0.4, linewidth=1.5, zorder=4)
                
                ax3.annotate(f'T{i+1}', (tornado[0], tornado[1]),
                           xytext=(5, -15), textcoords='offset points',
                           fontsize=8, color='black')
        
        ax3.set_title('Combined View\n(Territories + Connections)', 
                     fontsize=13, fontweight='bold')
        ax3.set_xlabel('X Coordinate (km)', fontsize=10)
        ax3.set_ylabel('Y Coordinate (km)', fontsize=10)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.set_aspect('equal')
        
        plt.suptitle(f'Tornado Shelter Analysis - {self.metric.capitalize()} Distance',
                    fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig


def main():
    """Main demonstration of the Tornado Shelter Allocation System."""
    print("TORNADO SHELTER ALLOCATION SYSTEM")
    print("Using Voronoi Diagrams and Delaunay Triangulations")
    print()
    
    N_SHELTERS = 8
    N_TORNADOES = 10
    BOUNDS = (0, 100, 0, 100)
    RANDOM_SEED = 42
    
    print(f"Configuration:")
    print(f"  Number of shelters: {N_SHELTERS}")
    print(f"  Number of tornado strikes: {N_TORNADOES}")
    print(f"  City bounds: {BOUNDS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()
    
    print("Generating random shelter locations...")
    shelters = TornadoShelterSystem.generate_random_shelters(
        N_SHELTERS, BOUNDS, seed=RANDOM_SEED
    )
    print(f"Generated {len(shelters)} shelters:")
    for i, shelter in enumerate(shelters):
        print(f"  Shelter {i}: ({shelter[0]:.2f}, {shelter[1]:.2f})")
    print()
    
    print("\nEXPERIMENT 0: Separate Views (Voronoi, Delaunay, Combined)")
    
    system_euclidean = TornadoShelterSystem(shelters, BOUNDS, metric='euclidean')
    tornado_results_euclidean = system_euclidean.simulate_tornado_strikes(
        N_TORNADOES, seed=RANDOM_SEED
    )
    
    fig0 = system_euclidean.plot_separate_views(tornado_results_euclidean)
    plt.savefig('tornado_shelter_separate_views.png', dpi=300, bbox_inches='tight')
    print("\nSaved: tornado_shelter_separate_views.png")
    
    print("\nEXPERIMENT 1: Euclidean Distance (L2 Norm)")
    
    fig1 = system_euclidean.plot_system(
        tornado_results_euclidean,
        title='Tornado Shelter Allocation - Euclidean Distance (L2 Norm)'
    )
    plt.savefig('tornado_shelter_euclidean.png', dpi=300, bbox_inches='tight')
    print("\nSaved: tornado_shelter_euclidean.png")
    
    print("\nEXPERIMENT 2: Manhattan Distance (L1 Norm)")
    
    system_manhattan = TornadoShelterSystem(shelters, BOUNDS, metric='manhattan')
    tornado_results_manhattan = system_manhattan.simulate_tornado_strikes(
        N_TORNADOES, seed=RANDOM_SEED
    )
    
    fig2 = system_manhattan.plot_system(
        tornado_results_manhattan,
        title='Tornado Shelter Allocation - Manhattan Distance (L1 Norm)'
    )
    plt.savefig('tornado_shelter_manhattan.png', dpi=300, bbox_inches='tight')
    print("\nSaved: tornado_shelter_manhattan.png")
    
    print("\nEXPERIMENT 3: Comparing Euclidean vs Manhattan Distance")
    
    tornado_points = np.array([r['tornado_point'] for r in tornado_results_euclidean])
    
    comparison_results = system_euclidean.compare_metrics(
        tornado_points, 
        metrics=['euclidean', 'manhattan']
    )
    fig3, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    for idx, (metric, data) in enumerate(comparison_results.items()):
        ax = axes[idx]
        system = data['system']
        results = data['results']
        
        xmin, xmax, ymin, ymax = BOUNDS
        ax.set_xlim(xmin - 5, xmax + 5)
        ax.set_ylim(ymin - 5, ymax + 5)
        
        # Plot Voronoi regions
        system._plot_voronoi_regions(ax)
        
        # Plot Delaunay triangulation
        system._plot_delaunay_triangulation(ax)
        
        # Plot shelters
        ax.scatter(system.shelters[:, 0], system.shelters[:, 1],
                  c='red', s=200, marker='o', edgecolors='darkred',
                  linewidths=2, label='Shelters', zorder=5)
        
        # Label shelters
        for i, shelter in enumerate(system.shelters):
            ax.annotate(f'S{i}', (shelter[0], shelter[1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='darkred')
        
        # Plot tornado strikes
        tornado_pts = np.array([r['tornado_point'] for r in results])
        ax.scatter(tornado_pts[:, 0], tornado_pts[:, 1],
                  c='black', s=150, marker='x', linewidths=3,
                  label='Tornado Strikes', zorder=6)
        
        # Draw lines from tornado to nearest shelter
        for i, result in enumerate(results):
            tornado = result['tornado_point']
            shelter = result['shelter_coords']
            ax.plot([tornado[0], shelter[0]], [tornado[1], shelter[1]],
                   'k--', alpha=0.4, linewidth=1, zorder=4)
        
        ax.set_title(f'{metric.capitalize()} Distance', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate (km)', fontsize=11)
        ax.set_ylabel('Y Coordinate (km)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
    
    plt.suptitle('Comparison: Euclidean vs Manhattan Distance Metrics',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('tornado_shelter_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: tornado_shelter_comparison.png")
    
    print("\nEXPERIMENT 4: Impact of Adding a New Shelter")
    
    new_shelter_location = np.array([[50, 50]])
    shelters_expanded = np.vstack([shelters, new_shelter_location])
    
    print(f"Adding new shelter at: ({new_shelter_location[0][0]:.2f}, {new_shelter_location[0][1]:.2f})")
    
    system_expanded = TornadoShelterSystem(shelters_expanded, BOUNDS, metric='euclidean')
    tornado_results_expanded = system_expanded.simulate_tornado_strikes(
        N_TORNADOES, seed=RANDOM_SEED
    )
    
    fig4 = system_expanded.plot_system(
        tornado_results_expanded,
        title=f'Tornado Shelter Allocation - With Additional Shelter (Total: {len(shelters_expanded)})'
    )
    plt.savefig('tornado_shelter_expanded.png', dpi=300, bbox_inches='tight')
    print("\nSaved: tornado_shelter_expanded.png")
    
    print("\nSUMMARY STATISTICS")
    
    for metric in ['euclidean', 'manhattan']:
        if metric == 'euclidean':
            results = tornado_results_euclidean
        else:
            results = tornado_results_manhattan
        
        distances = [r['distance'] for r in results]
        print(f"\n{metric.upper()} Distance:")
        print(f"  Average distance to shelter: {np.mean(distances):.2f} km")
        print(f"  Maximum distance to shelter: {np.max(distances):.2f} km")
        print(f"  Minimum distance to shelter: {np.min(distances):.2f} km")
        print(f"  Standard deviation: {np.std(distances):.2f} km")
    
    distances_expanded = [r['distance'] for r in tornado_results_expanded]
    print(f"\nWith ADDITIONAL SHELTER (Euclidean):")
    print(f"  Average distance to shelter: {np.mean(distances_expanded):.2f} km")
    print(f"  Maximum distance to shelter: {np.max(distances_expanded):.2f} km")
    print(f"  Improvement: {np.mean([r['distance'] for r in tornado_results_euclidean]) - np.mean(distances_expanded):.2f} km (average)")
    
    print("\nAll experiments completed successfully!")
    print("\nGenerated files:")
    print("  0. tornado_shelter_separate_views.png")
    print("  1. tornado_shelter_euclidean.png")
    print("  2. tornado_shelter_manhattan.png")
    print("  3. tornado_shelter_comparison.png")
    print("  4. tornado_shelter_expanded.png")
    
    plt.show()


if __name__ == "__main__":
    main()


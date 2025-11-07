"""
Demo Script - NO AUTO-SAVE
This version shows visualizations but does NOT save PNG files automatically.
You can manually save from the matplotlib window if needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from tornado_shelter_voronoi import TornadoShelterSystem

def main():
    print("THREE-VIEW DEMONSTRATION - NO AUTO-SAVE VERSION")
    print()
    
    N_SHELTERS = 8
    N_TORNADOES = 5
    BOUNDS = (0, 100, 0, 100)
    SEED = 42
    
    print(f"Configuration:")
    print(f"  - Shelters: {N_SHELTERS}")
    print(f"  - Tornado strikes: {N_TORNADOES}")
    print()
    
    print("Generating shelter locations...")
    shelters = TornadoShelterSystem.generate_random_shelters(
        N_SHELTERS, BOUNDS, seed=SEED
    )
    
    system = TornadoShelterSystem(shelters, BOUNDS, metric='euclidean')
    
    print("Simulating tornado strikes...")
    tornado_results = system.simulate_tornado_strikes(N_TORNADOES, seed=SEED)
    
    print("\nCreating visualization (NO files saved automatically)...")
    fig = system.plot_separate_views(tornado_results, figsize=(20, 7))
    
    print("\n[OK] Visualization ready!")
    print("     Close the window when done (no files will be saved)")
    print("     Or use File > Save from the matplotlib window to save manually")
    
    plt.show()


if __name__ == "__main__":
    main()


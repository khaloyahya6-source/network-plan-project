import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_manager import DataManager
from optimizer import RF_Optimizer
from neighbor_engine import NeighborEngine
from exporter import KMLExporter
from config import parse_tech_string
import os

def main():
    print("=== JULES V01: Autonomous Multi-Tech RF Optimization & ANR Engine ===")

    # 1. Data Ingestion
    dm = DataManager('towers.xlsx')
    towers = dm.load_towers()
    if towers is None:
        # Create a sample file if not exists for demo purposes
        print("Creating sample towers.xlsx...")
        sample_data = {
            'Tower_ID': ['Site_001', 'Site_002', 'Site_003'],
            'Lat': [33.5138, 33.5200, 33.5100],
            'Lon': [36.2765, 36.2850, 36.2900],
            'Tech_String': ['L1800', '4G2100', '5G_N78'],
            'Total_Height_m': [35, 40, 30]
        }
        pd.DataFrame(sample_data).to_excel('towers.xlsx', index=False)
        towers = dm.load_towers()

    # 2. GEE Ingestion
    if not dm.initialize_gee():
        grid_data = dm.get_dummy_data()
    else:
        grid_data = dm.fetch_geospatial_data()
        if grid_data is None:
            grid_data = dm.get_dummy_data()

    # 3. AI Optimization
    opt = RF_Optimizer(towers, grid_data)
    optimized_params = opt.run_optimization(n_particles=60, max_iter=100)

    # 4. Final Calculations
    rsrp_mats, sinr_mats = opt.get_final_matrices(optimized_params)

    # 5. ANR Engine
    neighbor_engine = NeighborEngine(towers, grid_data)
    neighbor_matrix = neighbor_engine.compute_neighbor_matrix(rsrp_mats, optimized_params)

    # 6. Export Results
    results = []
    for t_idx in range(len(towers)):
        tech, freq = parse_tech_string(towers.iloc[t_idx]['Tech_String'])
        for s_idx in range(3):
            results.append({
                'Tower_ID': towers.iloc[t_idx]['Tower_ID'],
                'Tech': tech,
                'Frequency': freq,
                'Sector': s_idx + 1,
                'Optimal_Azimuth': round(optimized_params[t_idx, s_idx, 0], 2),
                'Optimal_Tilt': round(optimized_params[t_idx, s_idx, 1], 2)
            })

    pd.DataFrame(results).to_excel('optimized_network_audit.xlsx', index=False)
    neighbor_matrix.to_excel('anr_neighbor_matrix.xlsx', index=False)
    print("Reports exported: optimized_network_audit.xlsx, anr_neighbor_matrix.xlsx")

    # 7. KML Export
    kml_exp = KMLExporter(opt.physics)
    kml_exp.create_kml(towers, optimized_params, 'network_plan.kml')

    # 8. Visualization
    print("Generating Network Heatmap...")
    # Composite RSRP Heatmap (Max RSRP over all sectors)
    max_rsrp = np.max(np.stack(rsrp_mats), axis=0)

    plt.figure(figsize=(12, 10))
    plt.imshow(max_rsrp, extent=[grid_data['bbox'][0], grid_data['bbox'][2], grid_data['bbox'][1], grid_data['bbox'][3]],
               cmap='jet', vmin=-115, vmax=-60)
    plt.colorbar(label='Max RSRP (dBm)')

    # Plot Towers and Orientations
    for t_idx, row in towers.iterrows():
        plt.plot(row['Lon'], row['Lat'], 'k^', markersize=10)
        plt.text(row['Lon'], row['Lat'], row['Tower_ID'], fontsize=9, fontweight='bold')

        # Draw arrows for sectors
        for s_idx in range(3):
            az = optimized_params[t_idx, s_idx, 0]
            # Convert azimuth to math angle
            math_az = np.radians(90 - az)
            dx = 0.005 * np.cos(math_az)
            dy = 0.005 * np.sin(math_az)
            plt.arrow(row['Lon'], row['Lat'], dx, dy, head_width=0.001, color='white')

    plt.title("JULES V01: Autonomous RF Optimization Heatmap")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig('network_composite_heatmap.png')
    print("Heatmap saved to network_composite_heatmap.png")

if __name__ == "__main__":
    main()

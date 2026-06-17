import numpy as np
import pandas as pd
from config import THRESHOLDS, parse_tech_string

class NeighborEngine:
    def __init__(self, towers_df, grid_data):
        self.towers_df = towers_df
        self.grid_data = grid_data
        self.scale = grid_data['scale']
        self.pixel_area = self.scale**2

    def compute_neighbor_matrix(self, rsrp_matrices, optimized_params):
        """
        rsrp_matrices: flat list of RSRP matrices (Tower1_S1, T1_S2, T1_S3, T2_S1...)
        optimized_params: (Num_Towers, 3, 2)
        """
        num_sectors = len(rsrp_matrices)
        neighbors = []

        rsrp_thresh = THRESHOLDS['RSRP']['Mid']
        area_thresh_m2 = 5000
        pixel_thresh = area_thresh_m2 / self.pixel_area

        print(f"Computing neighbors for {num_sectors} sectors...")

        for i in range(num_sectors):
            t_a_idx = i // 3
            s_a_idx = i % 3
            tech_a, freq_a = parse_tech_string(self.towers_df.iloc[t_a_idx]['Tech_String'])
            id_a = f"{self.towers_df.iloc[t_a_idx]['Tower_ID']}_{s_a_idx+1}"

            for j in range(i + 1, num_sectors):
                t_b_idx = j // 3
                s_b_idx = j % 3
                tech_b, freq_b = parse_tech_string(self.towers_df.iloc[t_b_idx]['Tech_String'])
                id_b = f"{self.towers_df.iloc[t_b_idx]['Tower_ID']}_{s_b_idx+1}"

                # Check overlap
                overlap_pixels = np.sum((rsrp_matrices[i] >= rsrp_thresh) & (rsrp_matrices[j] >= rsrp_thresh))

                if overlap_pixels >= pixel_thresh:
                    # Determine relation type
                    if tech_a == tech_b:
                        if freq_a == freq_b:
                            rel_type = "Intra-Frequency"
                        else:
                            rel_type = "Inter-Frequency"
                    else:
                        rel_type = "Inter-RAT"

                    overlap_size = overlap_pixels * self.pixel_area

                    neighbors.append({
                        'Source_Sector': id_a,
                        'Target_Sector': id_b,
                        'Relation_Type': rel_type,
                        'Overlap_Area_m2': round(overlap_size, 2),
                        'Tech_Source': tech_a,
                        'Tech_Target': tech_b
                    })

                    # Add symmetric relation
                    neighbors.append({
                        'Source_Sector': id_b,
                        'Target_Sector': id_a,
                        'Relation_Type': rel_type,
                        'Overlap_Area_m2': round(overlap_size, 2),
                        'Tech_Source': tech_b,
                        'Tech_Target': tech_a
                    })

        return pd.DataFrame(neighbors)

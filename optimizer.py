import numpy as np
from sko.PSO import PSO
from physics_engine import PhysicsEngine
from config import parse_tech_string, THRESHOLDS, MAX_TILT, MIN_TILT

class RF_Optimizer:
    def __init__(self, towers_df, grid_data):
        self.towers_df = towers_df
        self.grid_data = grid_data
        self.physics = PhysicsEngine(grid_data)
        self.num_towers = len(towers_df)
        self.tech_data = [parse_tech_string(row['Tech_String']) for _, row in towers_df.iterrows()]

    def objective_function(self, p):
        """
        p: flat array of [az1_1, tilt1_1, az1_2, tilt1_2, az1_3, tilt1_3, az2_1, ...]
        Optimization for 3 sectors per tower.
        """
        params = p.reshape((self.num_towers, 3, 2)) # (Tower, Sector, [Az, Tilt])

        # Group sectors by Tech/Frequency for Layered SINR analysis
        layers = {}
        for t_idx in range(self.num_towers):
            tech, freq = self.tech_data[t_idx]
            layer_key = f"{tech}_{freq}"
            if layer_key not in layers:
                layers[layer_key] = []

            tower_row = self.towers_df.iloc[t_idx]
            for s_idx in range(3):
                az = params[t_idx, s_idx, 0]
                tilt = params[t_idx, s_idx, 1]
                rsrp = self.physics.compute_rsrp(tower_row, az, tilt, freq)
                layers[layer_key].append((t_idx, s_idx, rsrp))

        # Compute Quality (SINR) and Strength (RSRP) per technology layer
        all_rsrp = [None] * (self.num_towers * 3)
        all_sinr = [None] * (self.num_towers * 3)

        total_sinr_fitness = 0
        total_interference_penalty = 0

        rsrp_thresh = THRESHOLDS['RSRP']['Mid'] # -95 dBm
        sinr_thresh = THRESHOLDS['SINR']['Target'] # 3 dB

        for layer_key, sectors in layers.items():
            layer_rsrps = [s[2] for s in sectors]
            layer_sinrs = self.physics.compute_global_sinr(layer_rsrps)

            # Interference Analysis: Find pixels with high overlap from DIFFERENT towers
            # Group by tower ID
            tower_best_rsrp = {}
            for i, (t_idx, s_idx, rsrp) in enumerate(sectors):
                flat_idx = t_idx * 3 + s_idx
                all_rsrp[flat_idx] = rsrp
                all_sinr[flat_idx] = layer_sinrs[i]

                # Quality score: Maximize area where SINR >= 3dB and RSRP >= -95dBm
                # We give more weight to quality (SINR)
                total_sinr_fitness += np.sum((rsrp >= rsrp_thresh) & (layer_sinrs[i] >= sinr_thresh))

                if t_idx not in tower_best_rsrp:
                    tower_best_rsrp[t_idx] = rsrp
                else:
                    tower_best_rsrp[t_idx] = np.maximum(tower_best_rsrp[t_idx], rsrp)

            # Savage Inter-Site Interference Penalty
            if len(tower_best_rsrp) > 1:
                tower_mats = list(tower_best_rsrp.values())
                stack = np.stack(tower_mats)
                # Count towers covering each pixel with strong signal
                count_map = np.sum(stack >= rsrp_thresh, axis=0)
                # Pixels with 2 or more strong interferers
                overlap_pixels = np.sum(count_map >= 2)
                # Exponential penalty to brutally steer PSO away from overlaps
                total_interference_penalty += (overlap_pixels ** 1.5) * 100

        # Geometric Constraints (Intra-site separation)
        intra_penalty = 0
        for t_idx in range(self.num_towers):
            for s1 in range(3):
                for s2 in range(s1 + 1, 3):
                    diff = abs(params[t_idx, s1, 0] - params[t_idx, s2, 0])
                    diff = min(diff, 360 - diff)
                    if diff < 45: # Standard 120 deg separation check
                        intra_penalty += (45 - diff) * 10000

        # Overshooting Penalty
        overshoot_penalty = 0
        for i, rsrp in enumerate(all_rsrp):
            if rsrp is None: continue
            t_idx = i // 3
            dist_m, _, _ = self.physics.calculate_vectors(self.towers_df.iloc[t_idx]['Lat'], self.towers_df.iloc[t_idx]['Lon'], self.towers_df.iloc[t_idx]['Total_Height_m'])
            overshoot_penalty += np.sum((rsrp >= -90) & (dist_m > 3000)) * 50

        # Final Fitness: Maximize Quality Area, Minimize Interference and Overlap
        fitness = total_sinr_fitness - total_interference_penalty - intra_penalty - overshoot_penalty
        return -fitness # PSO minimizes

    def run_optimization(self, n_particles=20, max_iter=30):
        dim = self.num_towers * 3 * 2
        lb = []
        ub = []
        for _ in range(self.num_towers):
            for _ in range(3):
                lb.extend([0, MIN_TILT])
                ub.extend([360, MAX_TILT])

        print(f"Executing PSO Engine ({dim}D space, {n_particles} particles)...")
        pso = PSO(func=self.objective_function, n_dim=dim, pop=n_particles, max_iter=max_iter, lb=lb, ub=ub, verbose=True)
        best_x, best_y = pso.run()

        optimized_params = best_x.reshape((self.num_towers, 3, 2))
        return optimized_params

    def get_final_matrices(self, optimized_params):
        layers = {}
        for t_idx in range(self.num_towers):
            tech, freq = self.tech_data[t_idx]
            layer_key = f"{tech}_{freq}"
            if layer_key not in layers:
                layers[layer_key] = []

            tower_row = self.towers_df.iloc[t_idx]
            for s_idx in range(3):
                az = optimized_params[t_idx, s_idx, 0]
                tilt = optimized_params[t_idx, s_idx, 1]
                rsrp = self.physics.compute_rsrp(tower_row, az, tilt, freq)
                layers[layer_key].append((t_idx, s_idx, rsrp))

        all_rsrp = [None] * (self.num_towers * 3)
        all_sinr = [None] * (self.num_towers * 3)

        for layer_key, sectors in layers.items():
            layer_rsrps = [s[2] for s in sectors]
            layer_sinrs = self.physics.compute_global_sinr(layer_rsrps)
            for i, (t_idx, s_idx, rsrp) in enumerate(sectors):
                flat_idx = t_idx * 3 + s_idx
                all_rsrp[flat_idx] = rsrp
                all_sinr[flat_idx] = layer_sinrs[i]

        return all_rsrp, all_sinr

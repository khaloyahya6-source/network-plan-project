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

        # Group sectors by Tech/Frequency for SINR (Interference only happens in-band)
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

        # Compute SINR per layer and collect results
        all_rsrp = [None] * (self.num_towers * 3)
        all_sinr = [None] * (self.num_towers * 3)

        for layer_key, sectors in layers.items():
            layer_rsrps = [s[2] for s in sectors]
            layer_sinrs = self.physics.compute_global_sinr(layer_rsrps)
            for i, (t_idx, s_idx, rsrp) in enumerate(sectors):
                flat_idx = t_idx * 3 + s_idx
                all_rsrp[flat_idx] = rsrp
                all_sinr[flat_idx] = layer_sinrs[i]

        # Calculate Fitness
        total_fitness = 0
        rsrp_thresh = THRESHOLDS['RSRP']['Mid']
        sinr_thresh = THRESHOLDS['SINR']['Target']

        for i, (rsrp, sinr) in enumerate(zip(all_rsrp, all_sinr)):
            # Good coverage pixels
            good_pixels = np.sum((rsrp >= rsrp_thresh) & (sinr >= sinr_thresh))
            total_fitness += good_pixels

            # Penalize Intra-site overlap
            t_idx = i // 3
            s_idx = i % 3
            for other_s in range(3):
                if s_idx == other_s: continue
                az_diff = abs(params[t_idx, s_idx, 0] - params[t_idx, other_s, 0])
                az_diff = min(az_diff, 360 - az_diff)
                if az_diff < 45: # Penalty for sectors too close
                    total_fitness -= 5000 * (45 - az_diff)

            # Penalize Overshooting
            dist_m, _, _ = self.physics.calculate_vectors(self.towers_df.iloc[t_idx]['Lat'], self.towers_df.iloc[t_idx]['Lon'], self.towers_df.iloc[t_idx]['Total_Height_m'])
            overshoot = np.sum((rsrp >= -90) & (dist_m > 3000))
            total_fitness -= overshoot * 10

        # Strict Inter-Site Interference Penalty
        # For each tech layer, check for pixels with strong RSRP from multiple towers
        for layer_key, sectors in layers.items():
            if len(sectors) <= 3: continue # Only check if multiple towers exist in this layer

            # Group rsrps by tower
            tower_rsrps = {}
            for t_idx, s_idx, rsrp in sectors:
                if t_idx not in tower_rsrps:
                    tower_rsrps[t_idx] = []
                tower_rsrps[t_idx].append(rsrp)

            if len(tower_rsrps) < 2: continue

            # Best RSRP per tower
            best_rsrp_per_tower = []
            for t_idx, rsrps in tower_rsrps.items():
                best_rsrp_per_tower.append(np.max(np.stack(rsrps), axis=0))

            # Penalty if top 2 towers both have RSRP > -95 in same pixel
            stack = np.stack(best_rsrp_per_tower)
            # Find pixels where at least 2 towers have RSRP > -95
            strong_coverage_count = np.sum(stack > -95, axis=0)
            overlap_pixels = np.sum(strong_coverage_count >= 2)
            total_fitness -= overlap_pixels * 50 # Heavy penalty for inter-site co-channel overlap

        return -total_fitness # Minimization

    def run_optimization(self, n_particles=20, max_iter=30):
        # 3 sectors per tower, 2 vars per sector (Az, Tilt)
        dim = self.num_towers * 3 * 2

        lb = []
        ub = []
        for _ in range(self.num_towers):
            for _ in range(3):
                lb.extend([0, MIN_TILT])
                ub.extend([360, MAX_TILT])

        print(f"Starting PSO Optimization with {dim} dimensions...")
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

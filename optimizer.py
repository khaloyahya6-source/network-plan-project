import numpy as np
from sko.PSO import PSO
from physics_engine import PhysicsEngine
from config import get_tech_specs, THRESHOLDS, MAX_TILT, MIN_TILT

class RF_Optimizer:
    def __init__(self, towers_df, grid_data):
        self.towers_df = towers_df
        self.grid_data = grid_data
        self.physics = PhysicsEngine(grid_data)
        self.num_towers = len(towers_df)

    def objective_function(self, p):
        params = p.reshape((self.num_towers, 3, 2))

        # Group by layer for interference
        layers = {}
        for t_idx in range(self.num_towers):
            tech_str = self.towers_df.iloc[t_idx]['Tech_String']
            specs = get_tech_specs(tech_str)
            layer_key = f"{specs['freq_mhz']}" # Interference happens on same freq

            if layer_key not in layers: layers[layer_key] = []

            tower_row = self.towers_df.iloc[t_idx]
            for s_idx in range(3):
                az = params[t_idx, s_idx, 0]
                tilt = params[t_idx, s_idx, 1]
                rsrp = self.physics.compute_rsrp(tower_row, az, tilt, tech_str)
                layers[layer_key].append({'t_idx': t_idx, 's_idx': s_idx, 'rsrp': rsrp})

        total_fitness = 0

        for freq, sectors in layers.items():
            rsrp_mats = [s['rsrp'] for s in sectors]
            sinr_mats = self.physics.compute_global_sinr(rsrp_mats)

            # Determine Dominant Server per pixel (Best Server)
            stack = np.stack(rsrp_mats)
            best_rsrp = np.max(stack, axis=0)
            best_idx = np.argmax(stack, axis=0)

            # Quality first: Maximize Area where SINR >= 3 and RSRP >= -100
            for i, s in enumerate(sectors):
                # Pixel is served by this sector only if it's the best server
                is_best = (best_idx == i)
                rsrp = rsrp_mats[i]
                sinr = sinr_mats[i]

                # Fitness += Serving area pixels meeting quality target
                # Penalize serving pixels that don't meet targets
                quality_area = np.sum(is_best & (rsrp >= -100) & (sinr >= 3))
                total_fitness += quality_area * 10

                # Penalty for pollution (strong overlap)
                # If this sector is NOT best server but has RSRP > -105, it pollutes
                pollution = np.sum((~is_best) & (rsrp >= -105))
                total_fitness -= pollution * 2

            # Inter-site Overlap Penalty (Brutal)
            tower_mats = {}
            for i, s in enumerate(sectors):
                tid = s['t_idx']
                if tid not in tower_mats: tower_mats[tid] = rsrp_mats[i]
                else: tower_mats[tid] = np.maximum(tower_mats[tid], rsrp_mats[i])

            if len(tower_mats) > 1:
                t_stack = np.stack(list(tower_mats.values()))
                strong_count = np.sum(t_stack >= -95, axis=0)
                overlap_pixels = np.sum(strong_count >= 2)
                total_fitness -= (overlap_pixels ** 1.5) * 50

        # Geometric Constraints
        for t_idx in range(self.num_towers):
            for s1 in range(3):
                for s2 in range(s1 + 1, 3):
                    diff = abs(params[t_idx, s1, 0] - params[t_idx, s2, 0])
                    diff = min(diff, 360 - diff)
                    if diff < 45: total_fitness -= (45 - diff) * 10000

        return -total_fitness

    def run_optimization(self, n_particles=60, max_iter=100):
        dim = self.num_towers * 3 * 2
        lb, ub = [], []
        for _ in range(self.num_towers):
            for _ in range(3):
                lb.extend([0, MIN_TILT]); ub.extend([360, MAX_TILT])

        pso = PSO(func=self.objective_function, n_dim=dim, pop=n_particles, max_iter=max_iter, lb=lb, ub=ub, verbose=True)
        best_x, _ = pso.run()
        return best_x.reshape((self.num_towers, 3, 2))

    def get_final_matrices(self, optimized_params):
        all_rsrp = []
        all_sinr = []
        # Calculate exactly as objective_function but return all
        for t_idx in range(self.num_towers):
            tech_str = self.towers_df.iloc[t_idx]['Tech_String']
            for s_idx in range(3):
                az = optimized_params[t_idx, s_idx, 0]
                tilt = optimized_params[t_idx, s_idx, 1]
                all_rsrp.append(self.physics.compute_rsrp(self.towers_df.iloc[t_idx], az, tilt, tech_str))

        # SINR per tech layer
        layers = {}
        for i, rsrp in enumerate(all_rsrp):
            t_idx = i // 3
            specs = get_tech_specs(self.towers_df.iloc[t_idx]['Tech_String'])
            fk = f"{specs['freq_mhz']}"
            if fk not in layers: layers[fk] = []
            layers[fk].append((i, rsrp))

        all_sinr = [None] * len(all_rsrp)
        for fk, data in layers.items():
            idxs = [d[0] for d in data]
            mats = [d[1] for d in data]
            sinrs = self.physics.compute_global_sinr(mats)
            for i, idx in enumerate(idxs):
                all_sinr[idx] = sinrs[i]

        return all_rsrp, all_sinr

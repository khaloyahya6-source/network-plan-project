import numpy as np

class SectorOptimizer:
    def __init__(self, beamwidth=65, min_gap=10):
        self.beamwidth = beamwidth
        self.min_gap = min_gap

    def get_sector_priority(self, priority_map, center_angle):
        half_bw = self.beamwidth / 2
        indices = []
        for i in np.arange(center_angle - half_bw, center_angle + half_bw):
            indices.append(int(i % 360))
        return np.sum(priority_map[indices])

    def optimize(self, priority_map, context):
        """
        Determine number of sectors and their azimuths.
        Rules:
        - Highway/Rural: 2 sectors
        - Urban: 3 sectors
        - Dense Urban/Big Village: 4 sectors
        """
        if context == "Highway":
            num_sectors = 2
        elif context == "Rural":
            num_sectors = 2
        elif context == "Urban":
            num_sectors = 3
        else: # Suburban or fallback
            num_sectors = 3
            
        # Optional upgrade to 4 if density is very high and distributed
        if num_sectors == 3 and np.mean(priority_map) > np.percentile(priority_map, 70) * 1.5:
             num_sectors = 4

        azimuths = []
        available_map = priority_map.copy()
        
        for _ in range(num_sectors):
            best_score = -1
            best_angle = 0
            
            # Search for best center angle
            for angle in range(360):
                # Check for overlap with existing sectors
                overlap = False
                for existing in azimuths:
                    diff = abs(angle - existing)
                    if diff > 180: diff = 360 - diff
                    if diff < (self.beamwidth + self.min_gap):
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                score = self.get_sector_priority(available_map, angle)
                if score > best_score:
                    best_score = score
                    best_angle = angle
            
            if best_score > 0:
                azimuths.append(best_angle)
                # "Subtract" covered priority to encourage spreading
                half_bw = self.beamwidth / 2
                for i in np.arange(best_angle - half_bw, best_angle + half_bw):
                    available_map[int(i % 360)] *= 0.1 # Heavily reduce
            else:
                # No more room or no more priority
                break
        
        return sorted(azimuths)

if __name__ == "__main__":
    from analyzer import EnvironmentAnalyzer
    analyzer = EnvironmentAnalyzer()
    p_map, ctx = analyzer.analyze_environment(33.5138, 36.2765)
    optimizer = SectorOptimizer()
    azimuths = optimizer.optimize(p_map, ctx)
    print(f"Optimal Azimuths for {ctx}: {azimuths}")

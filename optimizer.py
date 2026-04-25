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
        Returns: (azimuths, explanation)
        """
        if context == "Highway":
            target_sectors = 2
        elif context == "Rural":
            target_sectors = 2
        elif context == "Urban":
            target_sectors = 3
        else: # Suburban or fallback
            target_sectors = 3

        # Optional upgrade to 4 if density is very high and distributed
        if target_sectors == 3 and np.mean(priority_map) > np.percentile(priority_map, 70) * 1.5:
             target_sectors = 4

        azimuths = []
        explanation = ""
        available_map = priority_map.copy()

        for i in range(target_sectors):
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

            # Use a threshold for "meaningful" score.
            # If the score is too low, it means the area is already covered or empty.
            if best_score > 0.5: # Threshold can be adjusted
                azimuths.append(best_angle)
                # "Subtract" covered priority to encourage spreading
                half_bw = self.beamwidth / 2
                for i in np.arange(best_angle - half_bw, best_angle + half_bw):
                    available_map[int(i % 360)] *= 0.1 # Heavily reduce
            else:
                if len(azimuths) < target_sectors:
                    explanation = f"Reduced from {target_sectors} to {len(azimuths)} sectors because remaining areas are already covered by other towers or have no priority targets."
                break

        return sorted(azimuths), explanation

if __name__ == "__main__":
    from analyzer import EnvironmentAnalyzer
    analyzer = EnvironmentAnalyzer()
    p_map, ctx = analyzer.analyze_environment(33.5138, 36.2765)
    optimizer = SectorOptimizer()
    azimuths = optimizer.optimize(p_map, ctx)
    print(f"Optimal Azimuths for {ctx}: {azimuths}")

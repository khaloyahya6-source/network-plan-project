import pandas as pd
import numpy as np
import ee
import math
from config import GEE_SCALE_LIMIT, BUFFER_KM, parse_tech_string

class DataManager:
    def __init__(self, excel_path='towers.xlsx'):
        self.excel_path = excel_path
        self.towers_df = None
        self.grid_data = {}

    def load_towers(self):
        print(f"Loading towers from {self.excel_path}...")
        try:
            df = pd.read_excel(self.excel_path)
            # Dynamic column mapping/cleaning
            expected = {
                'Tower_ID': ['Tower_ID', 'ID', 'SiteID', 'Site_ID'],
                'Lat': ['Lat', 'Latitude', 'Y'],
                'Lon': ['Lon', 'Longitude', 'X'],
                'Tech_String': ['Tech_String', 'Technology', 'Band', 'Tech'],
                'Total_Height_m': ['Total_Height_m', 'Height', 'Antenna_Height']
            }

            clean_df = pd.DataFrame()
            for col, aliases in expected.items():
                found = False
                for alias in aliases:
                    if alias in df.columns:
                        clean_df[col] = df[alias]
                        found = True
                        break
                if not found:
                    if col == 'Total_Height_m': clean_df[col] = 30 # Default
                    else: raise ValueError(f"Missing required column: {col}")

            self.towers_df = clean_df
            print(f"Successfully loaded {len(self.towers_df)} towers.")
            return self.towers_df
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return None

    def initialize_gee(self):
        try:
            ee.Initialize()
            print("Earth Engine initialized successfully.")
            return True
        except Exception as e:
            print(f"GEE Initialization failed: {e}. Ensure 'earthengine authenticate' has been run.")
            return False

    def fetch_geospatial_data(self):
        if self.towers_df is None: return None

        # 1. Bounding Box
        min_lat, max_lat = self.towers_df['Lat'].min(), self.towers_df['Lat'].max()
        min_lon, max_lon = self.towers_df['Lon'].min(), self.towers_df['Lon'].max()

        # Add 5km buffer (approx 0.045 degrees)
        buffer = BUFFER_KM / 111.0
        bbox = [min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer]
        region = ee.Geometry.Rectangle(bbox)

        # 2. Adaptive Resolution Scaling
        # Calculate area in degrees
        width_deg = (max_lon - min_lon) + 2*buffer
        height_deg = (max_lat - min_lat) + 2*buffer

        # Area in meters (approx)
        width_m = width_deg * 111000 * math.cos(math.radians(min_lat))
        height_m = height_deg * 111000

        total_area_m2 = width_m * height_m
        # pixels = total_area / (scale^2) <= GEE_SCALE_LIMIT
        # scale = sqrt(total_area / GEE_SCALE_LIMIT)
        min_scale = math.sqrt(total_area_m2 / GEE_SCALE_LIMIT)
        scale = max(10, min_scale) # Never go below 10m
        scale = min(90, scale)   # Max 90m
        print(f"Adaptive scale set to: {scale:.2f}m")

        # 3. Data Extraction
        # ESA WorldCover v200
        wc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(region)
        # Built-up is class 50
        built_up = wc.eq(50)

        # Dynamic Clutter Density (Convolutional approach via reduceNeighborhood)
        # 500m radius sliding window for density
        kernel_size_px = int(500 / scale)
        density = built_up.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.square(kernel_size_px)
        ).rename('density')

        # SRTM DEM
        dem = ee.Image('CGIAR/SRTM/Version4').clip(region)

        # Combine
        combined = ee.Image.cat([dem, density, wc])

        # Sample pixels
        # getRegion is better for small areas, but for larger we use sampleRectangle
        # though sampleRectangle has limits. We'll use sampleRectangle.
        try:
            data = combined.sampleRectangle(region=region, scale=scale)

            elev = np.array(data.get('elevation').getInfo())
            clutter_density = np.array(data.get('density').getInfo())
            land_cover = np.array(data.get('Map').getInfo())

            # Meshgrid for coordinates
            rows, cols = elev.shape
            lons = np.linspace(bbox[0], bbox[2], cols)
            lats = np.linspace(bbox[3], bbox[1], rows) # North to South
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            self.grid_data = {
                'elevation': elev,
                'clutter_density': clutter_density,
                'land_cover': land_cover,
                'lon_grid': lon_grid,
                'lat_grid': lat_grid,
                'scale': scale,
                'bbox': bbox
            }
            return self.grid_data
        except Exception as e:
            print(f"Error fetching GEE data: {e}")
            # Fallback to dummy data for development if needed,
            # but for production we return None or raise.
            return None

    def get_dummy_data(self):
        """Helper for local testing without GEE auth."""
        print("Generating dummy geospatial data for testing...")
        rows, cols = 100, 100
        min_lat, max_lat = self.towers_df['Lat'].min() - 0.05, self.towers_df['Lat'].max() + 0.05
        min_lon, max_lon = self.towers_df['Lon'].min() - 0.05, self.towers_df['Lon'].max() + 0.05

        lons = np.linspace(min_lon, max_lon, cols)
        lats = np.linspace(max_lat, min_lat, rows)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        self.grid_data = {
            'elevation': np.zeros((rows, cols)),
            'clutter_density': np.random.rand(rows, cols) * 0.5, # Some random clutter
            'land_cover': np.ones((rows, cols)) * 10, # Trees/Grass
            'lon_grid': lon_grid,
            'lat_grid': lat_grid,
            'scale': 30,
            'bbox': [min_lon, min_lat, max_lon, max_lat]
        }
        return self.grid_data

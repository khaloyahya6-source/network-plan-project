import re

# Technology Mapping Matrix
TECH_MAPPING = {
    'GSM': {'regex': r'(2G|G|GSM)', 'freq_default': 900},
    'UMTS': {'regex': r'(3G|U|WCDMA)', 'freq_default': 2100},
    'LTE': {'regex': r'(4G|L|LTE)', 'freq_default': 1800},
    'NR': {'regex': r'(5G|N|NR)', 'freq_default': 3500}
}

FREQ_MAP = {
    '900': 900, '1800': 1800, '2100': 2100, '2600': 2600,
    '700': 700, '800': 800, '850': 850, '1900': 1900,
    'N78': 3500, 'N41': 2500, 'N258': 26000,
    'U9': 900, 'L18': 1800, 'L21': 2100, 'L26': 2600
}

def parse_tech_string(tech_str):
    tech_str = str(tech_str).upper()
    tech = "LTE" # Default
    freq = 1800 # Default

    for t, data in TECH_MAPPING.items():
        if re.search(data['regex'], tech_str):
            tech = t
            freq = data['freq_default']
            break

    # Try to find exact frequency in string
    for key, f in FREQ_MAP.items():
        if key in tech_str:
            freq = f
            break

    # Fallback for numbers if no key matches
    nums = re.findall(r'\d+', tech_str)
    if nums:
        # If it's a small number like 9, it might be 900
        n = int(nums[0])
        if n < 100:
            if n == 9: freq = 900
            elif n == 18: freq = 1800
            elif n == 21: freq = 2100
            elif n == 26: freq = 2600
        else:
            freq = n

    return tech, freq

# RF Parameters Dictionary per Frequency (MHz)
RF_PARAMS = {
    'Low': {
        'tx_power_dbm': 46,
        'antenna_gain_dbi': 15,
        'hbw': 65,
        'vbw': 10,
        'front_to_back': 25
    },
    'Mid': {
        'tx_power_dbm': 46,
        'antenna_gain_dbi': 18,
        'hbw': 65,
        'vbw': 7,
        'front_to_back': 30
    },
    'High': {
        'tx_power_dbm': 43,
        'antenna_gain_dbi': 22,
        'hbw': 65,
        'vbw': 5,
        'front_to_back': 30
    }
}

def get_rf_params(freq_mhz):
    if freq_mhz < 1000:
        return RF_PARAMS['Low']
    elif freq_mhz < 3000:
        return RF_PARAMS['Mid']
    else:
        return RF_PARAMS['High']

# Propagation Thresholds
THRESHOLDS = {
    'RSRP': {
        'Excellent': -85,
        'Mid': -95,
        'CellEdge': -110
    },
    'SINR': {
        'Target': 3,
        'NoiseFloor': -114 # dBm for 20MHz
    }
}

# GEE Config
GEE_SCALE_LIMIT = 262144 # 512*512
BUFFER_KM = 5
MAX_TILT = 15
MIN_TILT = 0

import re

# Technology Mapping Matrix & Link Budget Specs
TECH_SPECS = {
    'GSM': {
        'regex': r'(2G|G|GSM)',
        'freq_mhz': 900,
        'tx_power_dbm': 43.0,
        'ant_gain_dbi': 15.0,
        'penetration_loss_db': 10.0,
        'rsrp_threshold': -100.0,
        'hbw': 65,
        'vbw': 10
    },
    'UMTS': {
        'regex': r'(3G|U|WCDMA)',
        'freq_mhz': 2100,
        'tx_power_dbm': 43.0,
        'ant_gain_dbi': 18.0,
        'penetration_loss_db': 15.0,
        'rsrp_threshold': -100.0,
        'hbw': 65,
        'vbw': 7
    },
    'LTE': {
        'regex': r'(4G|L|LTE)',
        'freq_mhz': 1800,
        'tx_power_dbm': 46.0,
        'ant_gain_dbi': 18.0,
        'penetration_loss_db': 15.0,
        'rsrp_threshold': -100.0,
        'hbw': 65,
        'vbw': 7
    },
    'NR': {
        'regex': r'(5G|N|NR)',
        'freq_mhz': 3500,
        'tx_power_dbm': 43.0,
        'ant_gain_dbi': 22.0,
        'penetration_loss_db': 25.0,
        'rsrp_threshold': -100.0,
        'hbw': 65,
        'vbw': 5
    }
}

def get_tech_specs(tech_str):
    tech_str = str(tech_str).upper()
    for tech, specs in TECH_SPECS.items():
        if re.search(specs['regex'], tech_str):
            return specs
    return TECH_SPECS['LTE'] # Default

def parse_tech_string(tech_str):
    specs = get_tech_specs(tech_str)
    # Find which tech key matches
    for tech, s in TECH_SPECS.items():
        if s == specs:
            return tech, s['freq_mhz']
    return "LTE", 1800

# Professional Propagation Thresholds
THRESHOLDS = {
    'RSRP': {
        'Excellent': -80,
        'Mid': -95,
        'CellEdge': -105
    },
    'SINR': {
        'Target': 3,
        'NoiseFloor': -114
    }
}

# GEE Config
GEE_SCALE_LIMIT = 262144
BUFFER_KM = 5
MAX_TILT = 15
MIN_TILT = 0
METERS_PER_DEGREE = 111320

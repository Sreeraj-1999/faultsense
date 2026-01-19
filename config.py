# Database
DATABASE_URL = "postgresql://memphis:Memphis%401234%24@192.168.18.175/obanext_8"  #was obanext5

# File paths
CSV_PATH_COMBINED = r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv"
CSV_PATH_TRAINING = r"C:\Users\User\Desktop\siemens\freya_schulte\training_data_averaged.csv"

# Model parameters
LOOKBACK = 672
HORIZON_SEQUENCE = 96
HORIZON_METADATA = 24
STEP = 12

# Training
MI_THRESHOLD = 0.05
TOP_N_CORRELATIONS = 25

# Server
SERVER_HOST = "192.168.18.233"
SERVER_PORT = 8000
import os

# --- PATH CONFIGURATION ---
# Get the absolute path to the project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This is where we want COCO and PeopleArt to live
# e.g., C:\Users\Lenovo\PycharmProjects\Replication Study\datasets
BASE_DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')

# Specific path for our processed PeopleArt
DATASET_DIR = os.path.join(BASE_DATASETS_DIR, 'PeopleArt_Replication')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'raw_people_art')
STYLE_LIST_DIR = os.path.join(DATASET_DIR, 'style_lists')

# --- MODEL SETTINGS ---
MODEL_TYPE = 'yolov8n.pt'
MODEL_NAME ='trained_art_model'
IMG_SIZE = 640
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 16

# --- SCIENTIFIC BASELINES ---
WESTLAKE_2016_BASELINE = 0.45

# --- HYPOTHESIS GROUPINGS ---
ABSTRACTION_MAP = {
    'High_Abstraction': [
        'Cubism', 'Expressionism', 'Impressionism', 'Pop Art', 'Modern Art'
    ],
    'Low_Abstraction': [
        'Baroque', 'Renaissance', 'Realism', 'Neoclassicism', 'Romanticism', 'Dutch Golden Age'
    ]
}
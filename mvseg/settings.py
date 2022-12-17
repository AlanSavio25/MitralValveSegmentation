from pathlib import Path

root = Path(__file__).parent #.parent  # top-level directory
DATA_PATH = '/cluster/project/infk/cvg/students/alpaul/MitralValveSegmentation/data/'
DATASETS_PATH = root / 'mvseg' / 'datasets/'  # datasets and pretrained weights
TRAINING_PATH = '/cluster/project/infk/cvg/students/alpaul/MitralValveSegmentation/outputs/training/'
EVAL_PATH = '/cluster/project/infk/cvg/students/alpaul/MitralValveSegmentation/outputs/results/'  # evaluation results

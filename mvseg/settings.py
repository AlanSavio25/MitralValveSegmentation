from pathlib import Path

root = Path(__file__).parent #.parent  # top-level directory
print(root)
DATA_PATH = '/Users/alansavio/ETHZ/Semester3/AML/task3/data/'
DATASETS_PATH = root / 'mvseg' / 'datasets/'  # datasets and pretrained weights
TRAINING_PATH = '/Users/alansavio/Semester3/AML/task3/outputs/training/'  # training checkpoints
EVAL_PATH = '/Users/alansavio/Semester3/AML/task3/outputs/results/'  # evaluation results

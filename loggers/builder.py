from .classification import ClassificationCSVLogger

def build_logger(csv_path, task):
    if task.lower() == 'classification':
        return ClassificationCSVLogger(csv_path)
    elif task.lower() == 'segmentation':
        raise NotImplementedError
    else:
        raise AssertionError(f"No such task! (task: {task})")

def build_visualizer():
    pass
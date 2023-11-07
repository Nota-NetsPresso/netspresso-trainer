from .register import POSTPROCESSOR_DICT


def build_postprocessor(task: str, conf_model):
    head_name = conf_model.architecture.head.name
    if head_name not in POSTPROCESSOR_DICT:
        return None
    return POSTPROCESSOR_DICT[head_name]()

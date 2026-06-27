import argparse

import torch
from torch import nn

from utils.model import Model


def _strip_module_prefix(state_dict):
    return {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }


def load_model(model_path, ssl_path, device):
    args = argparse.Namespace()
    model = Model(args, device, ssl_path=ssl_path)
    state_dict = torch.load(model_path, map_location=device)
    has_module_prefix = any(key.startswith("module.") for key in state_dict)

    use_cuda = str(device).startswith("cuda")
    if use_cuda:
        model = nn.DataParallel(model).to(device)
        if has_module_prefix:
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
    else:
        if has_module_prefix:
            state_dict = _strip_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        model = model.to(device)

    model.eval()
    return model

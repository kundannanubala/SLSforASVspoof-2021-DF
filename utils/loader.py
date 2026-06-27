import argparse

import torch
from torch import nn

from utils.model import Model


def load_model(model_path, ssl_path, device):
    args = argparse.Namespace()
    model = Model(args, device, ssl_path=ssl_path)
    state_dict = torch.load(model_path, map_location=device)
    model = nn.DataParallel(model).to(device)

    has_module_prefix = any(key.startswith("module.") for key in state_dict)
    if has_module_prefix:
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    model.eval()
    return model

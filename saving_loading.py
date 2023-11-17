from pathlib import Path
import os
import torch

def saving_model(model, model_name, saving_path):
    model_path = Path(saving_path)
    model_path = os.mkdir(parent = True, exist_ok = True)

    model_save_path = model_path / model_name
    torch.save(obj = model.state_dict(), f = model_save_path)

def loading_model(model, model_save_path, device):
    new_model = model
    new_model.load_state_dict(torch.load(d = model_save_path))
    new_model.to(device)
    return new_model

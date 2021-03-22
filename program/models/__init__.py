import os, sys, time, random, importlib
import torch

class basic_model(torch.nn.Module):
    def __init__(self, model_name, embed_dim, hidden_dim):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        print('Model [{}] has been built.'.format(self.model_name))

model_dict = {}

def load_model(model_name):
    return model_dict[model_name].setup_model()

def register_model(name):
    def register_model(cls):
        if name in model_dict:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        model_dict[name] = cls
        return cls
    return register_model

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('program.models.' + module)


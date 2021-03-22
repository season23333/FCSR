import os, sys, torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from .. import datasets_utils, basic_dataset
from ... import modules

class basic_label_dataset(basic_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.task_type = 'c'
    
    def load_dataset(self, file_trn: str, file_val: str = None, file_tst: str = None):
        self.dict = self.build_dict([file_trn, file_val, file_tst])
        self.basic_load_dataset(file_trn, file_val, file_tst)

        if isinstance(self.dict, list):
            self.embedding_matrix = [self.load_embedding(self.dict[0]), self.load_embedding(self.dict[1])]
        else:
            self.embedding_matrix = self.load_embedding(self.dict)
    

    
    





















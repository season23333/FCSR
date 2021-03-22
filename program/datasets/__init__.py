import os, sys, time, random, importlib, torch, math
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from . import datasets_utils

dataset_dict = {}
dataset_names = ['datasets_classification']
supported_transformers = ['bert-base-uncased']

class tvt_dataset(torch.utils.data.Dataset):
    def __init__(self, source, wizard, target):
        super().__init__()
        self.source = source
        self.wizard = wizard
        self.target = target

    def __getitem__(self, index):
        src = self.source[index]
        wiz = self.wizard[index] if self.wizard else torch.tensor(0)
        tgt = self.target[index]
        return [src, wiz, tgt]

    def __len__(self):
        return len(self.target)

class basic_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config):
        self.config = dataset_config

        self.transformers_dict = [
            'bert-base-uncased',
            'bert-large-uncased',

        ]

        self.glove_dict = [
            'glove.6B.50d',
            'glove.6B.100d',
            'glove.6B.200d',
            'glove.6B.300d',
            'glove.840B.300d',
        ]

        self.character_dict = ['glove.840B.300d-char']

    def basic_load_dataset(self, file_trn: str, file_val: str, file_tst: str): # 返回划分好的数据集 np(np)
        # Load_file: [np.array source], [np.array wizard], [np.array target]
        swt_trn = self.load_file(file_trn) # train file must exists
        swt_val = self.load_file(file_val) if file_val is not None else [None, None, None]
        swt_tst = self.load_file(file_tst) if file_tst is not None else [None, None, None]

        source, wizard, target = self.permute(swt_trn, swt_val, swt_tst)
        if hasattr(self, 'balance') and self.balance: source, wizard, target = self.process_balance(source, wizard, target)

        source_trn  = source[:-(self.val_num + self.tst_num)]
        wizard_trn  = wizard[:-(self.val_num + self.tst_num)] if wizard is not None else None
        target_trn  = target[:-(self.val_num + self.tst_num)]

        source_val  = source[-(self.val_num + self.tst_num) : -self.tst_num]
        wizard_val  = wizard[-(self.val_num + self.tst_num) : -self.tst_num] if wizard is not None else None
        target_val  = target[-(self.val_num + self.tst_num) : -self.tst_num]

        source_tst  = source[-self.tst_num:]
        wizard_tst  = wizard[-self.tst_num:] if wizard is not None else None
        target_tst  = target[-self.tst_num:]

        self.src_trn_len, self.src_val_len, self.src_tst_len = self.avg_len([source_trn, source_val, source_tst])
        self.wiz_trn_len, self.wiz_val_len, self.wiz_tst_len = self.avg_len([wizard_trn, wizard_val, wizard_tst]) if wizard_trn is not None else [0, 0, 0]
        self.tgt_trn_dic, self.tgt_val_dic, self.tgt_tst_dic = self.target_statisic([target_trn, target_val, target_tst]) if self.task_type in ['c'] else [0, 0, 0]
        self.target_set_trn, self.target_set_val, self.target_set_tst = set(target_trn), set(target_val), set(target_tst) if self.task_type in ['c'] else [0, 0, 0]
        self.trn_num, self.val_num, self.tst_num = len(source_trn), len(source_val), len(source_tst)

        self.data = {
            'trn': tvt_dataset(source_trn, wizard_trn, target_trn),
            'val': tvt_dataset(source_val, wizard_val, target_val),
            'tst': tvt_dataset(source_tst, wizard_tst, target_tst)
        }
    
    def load_embedding(self, dictionary):
        if self.config.pretrained is None or self.config.pretrained == 'character':
            embedding_matrix = None
        elif 'glove' in self.config.pretrained:
            embedding_matrix = datasets_utils.get_embedding_matrix(self.config.embed_dim, dictionary, self.config.pretrained)
        elif self.config.pretrained in supported_transformers:
            embedding_matrix = self.config.pretrained
        else:
           raise ValueError(f'Unknown pretrained operation [{self.config.pretrained}].')

        return embedding_matrix

    def permute(self, swt_trn, swt_val, swt_tst):
        # list[np.array]
        source_trn, wizard_trn, target_trn = swt_trn
        source_val, wizard_val, target_val = swt_val# if swt_val is not None else None
        source_tst, wizard_tst, target_tst = swt_tst# if swt_tst is not None else None

        source = source_trn
        if source_val is not None: source += source_val
        if source_tst is not None: source += source_tst

        wizard = wizard_trn
        if wizard_val is not None: wizard += wizard_val
        if wizard_tst is not None: wizard += wizard_tst

        target = target_trn
        if target_val is not None: target += target_val
        if target_tst is not None: target += target_tst

        return source, wizard, target
    
    def process_balance(self, source, wizard, target):
        print(f'Balance mode will probably disturb the order of original dataset.')
        lable_kind = set(target)
        src_dict = {}
        wiz_dict = None if wizard is None or len(wizard) == 0 else {}
        cnt_dict = {}
        total = len(target)
        for i in lable_kind:
            src_dict[i] = []
            if wiz_dict is not None: wiz_dict[i] = []
            cnt_dict[i] = 0
        for idx, tgt in enumerate(target):
            src_dict[tgt].append(source[idx])
            if wiz_dict is not None: wiz_dict[tgt].append(wizard[idx])
            cnt_dict[tgt] += 1

        trn_nums = {}
        val_nums = {}
        tst_nums = {}
        for k, v in cnt_dict.items():
            val_nums[k] = math.floor(cnt_dict[k] // 10)
            tst_nums[k] = int(val_nums[k] * 2)
            trn_nums[k] = cnt_dict[k] - val_nums[k] - tst_nums[k]

        source_ret = []
        wizard_ret = None if wizard is None or len(wizard) == 0 else []
        target_ret = []
        for k, v in sorted(cnt_dict.items(), key = lambda kv:(kv[1], kv[0])):
            source_ret += src_dict[k][:trn_nums[k]]
            if wizard is not None: wizard_ret += wiz_dict[k][:trn_nums[k]]
            target_ret += [k] * trn_nums[k]
        
        for k, v in sorted(cnt_dict.items(), key = lambda kv:(kv[1], kv[0])):
            source_ret += src_dict[k][trn_nums[k]: -tst_nums[k]]
            if wizard is not None: wizard_ret += wiz_dict[k][trn_nums[k]: -tst_nums[k]]
            target_ret += [k] * val_nums[k]

        for k, v in sorted(cnt_dict.items(), key = lambda kv:(kv[1], kv[0])):
            source_ret += src_dict[k][-tst_nums[k]:]
            if wizard is not None: wizard_ret += wiz_dict[k][-tst_nums[k]:]
            target_ret += [k] * tst_nums[k]

        assert len(source_ret) == len(source), f'source len error, got source_ret = {len(source_ret)} and source = {len(source)}'
        assert len(target_ret) == len(target), f'source len error, got target_ret = {len(target_ret)} and target = {len(target)}'
        return source_ret, wizard_ret, target_ret

    def avg_len(self, matrices):
        ret = []
        for matrix in matrices:
            tmp_len = [len(item) for item in matrix]
            ret.append(sum(tmp_len) / len(tmp_len))
        return ret

    def target_statisic(self, targets):
        ret = []
        for target_vec in targets:
            tmp_target = {}
            for item in target_vec:
                if not item in tmp_target:
                    tmp_target[item] = 0
                tmp_target[item] += 1
            ret.append(tmp_target)
        return ret

    def print_self(self):
        if isinstance(self.dict, datasets_utils.Dictionary):
            print(f' The length of dictionary is{len(self.dict)}')
        elif isinstance(self, list):
            print(f' The length of dictionary is{len(self.dict[0])} and {len(dict[1])}')
        else:
            print(f'No dict found')
        print(f'Train num = {self.trn_num}, valid num = {self.val_num}, test num = {self.tst_num}')

        print(f'Average length of training source = {self.src_trn_len}')
        print(f'Average length of validation source = {self.src_val_len}')
        print(f'Average length of test source = {self.src_tst_len}')

        print(f'Average length of training wizard = {self.wiz_trn_len}')
        print(f'Average length of validation wizard = {self.wiz_trn_len}')
        print(f'Average length of test wizard = {self.wiz_trn_len}')

        if self.task_type in ['c']:
            print(f'Training targets = {self.tgt_trn_dic}')
            print(f'Validation targets = {self.tgt_val_dic}')
            print(f'Test targets = {self.tgt_tst_dic}')
            print(f'Class Index_trn = {self.target_set_trn}')
            print(f'Class Index_val = {self.target_set_val}')
            print(f'Class Index_tst = {self.target_set_tst}')
    
    def pretrained_type(self):
        if self.config.pretrained is None or self.config.pretrained in self.glove_dict:
            ret = 'index_word' # defalut
        elif self.config.pretrained in self.transformers_dict:
            ret = 'spec' # load from pretrained transformers
        elif self.config.pretrained in self.character_dict or self.config.pretrained == 'character':
            ret = 'index_char'
        else:
            raise ValueError(f'Unsupported pretrained mode [{self.config.pretrained}]')
        return ret

    def build_dict(self, file_names):
        pretrain_type = self.pretrained_type()

        if pretrain_type == 'index_word':
            ret = self.build_dict_index
        elif pretrain_type == 'index_char':
            ret = self.build_dict_char
        else:
            ret = self.build_dict_spec

        return ret(file_names)

    def load_file(self, 
                  filename, 
                  spliter = ' ', 
                  pre_eos = True, 
                  end_eos = True,
                  verbose = True,
                  lower = True,
                  remove_punc = True,
        ):
        pretrain_type = self.pretrained_type()

        if pretrain_type == 'index_word':
            ret = self.load_file_index
        elif pretrain_type == 'index_char':
            ret = self.load_file_char
        else:
            ret = self.load_file_spec

        return ret(filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc)
# ---------------------------------------------------------
def load_dataset(data_name):
    return dataset_dict[data_name].setup_dataset()

def register_dataset(name):
    def register_dataset(cls):
        if name in dataset_dict:
            raise ValueError('Cannot register duplicate dataset ({})'.format(name))
        dataset_dict[name] = cls
        return cls
    return register_dataset

for dirname in dataset_names:
    for file in os.listdir(os.path.dirname(__file__ ) + '/' + dirname):
        if file.endswith('.py') and not file.startswith('_'):
            module = file[:file.find('.py')]
            importlib.import_module('program.datasets.{}.'.format(dirname) + module)






    























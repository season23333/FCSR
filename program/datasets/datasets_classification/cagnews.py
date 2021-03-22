import os, sys, time, random, re, json, spacy, torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from spacy.tokenizer import Tokenizer
# from enchant.checker import SpellChecker
from .. import datasets_utils
from .. import register_dataset
from . import basic_label_dataset

@register_dataset('cagnews')
class cagnews(basic_label_dataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        data_dir = 'datasets/datasets_multi/AGnews'
        self.encoding = 'utf-8'
        self.num_label = 4
        self.dict_label = {1: 0, 2: 1, 3: 2, 4: 3}
        self.trn_path = os.path.join(data_dir, 'train.jsonl')
        self.val_path = os.path.join(data_dir, 'dev.jsonl')
        self.tst_path = os.path.join(data_dir, 'test.jsonl')

        self.val_num = 5000
        self.tst_num = 7600
        
        self.load_dataset(self.trn_path, self.val_path, self.tst_path)

    def load_file_spec(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        tokenizer = datasets_utils.transformer_tokenizer(self.config.pretrained)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                if ': null,' in line: line = re.sub(': null,', ': \"null\",', line)
                dt = json.loads(line)
                tmp_line = dt['headline'] + ' | ' + dt['text']

                indices = tokenizer(tmp_line)
                source.append(indices)
                target.append(self.dict_label[dt['label']])
        return source, wizard, target

    def load_file_index(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                if ': null,' in line: line = re.sub(': null,', ': \"null\",', line)
                dt = json.loads(line)

                wordss = datasets_utils.tokenize(dt['text'], spliter, pre_eos, end_eos, lower, remove_punc)
                wordsw = datasets_utils.tokenize(dt['headline'], spliter, pre_eos, end_eos, lower, remove_punc)
                indexs = [self.dict.get_index(word, True) for word in wordss]
                indexw = [self.dict.get_index(word, True) for word in wordsw]
                source.append(torch.tensor(indexs + indexw))
                target.append(self.dict_label[dt['label']])
        return source, wizard, target
    
    def load_file_char(self, filename, spliter, pre_eos, end_eos, verbose, lower, remove_punc):
        assert os.path.exists(filename), 'file [{}] does not exists.'.format(filename)
        source = []
        wizard = None
        target = []
        with open(filename, "r", encoding = self.encoding) as f:
            lines = tqdm(f.readlines(), ascii = True)
            for line in lines:
                lines.set_description(f"Processing {filename.split('/')[-1]}")
                if ': null,' in line: line = re.sub(': null,', ': \"null\",', line)
                dt = json.loads(line)

                wordss = datasets_utils.tokenize(dt['text'], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                wordsw = datasets_utils.tokenize(dt['headline'], spliter, pre_eos, end_eos, lower, remove_punc, character = True)
                indexs = [self.dict.get_index(word, True) for word in wordss]
                indexw = [self.dict.get_index(word, True) for word in wordsw]
                source.append(torch.tensor(indexs + indexw))
                target.append(self.dict_label[dt['label']])
        return source, wizard, target

    def build_dict_spec(self, file_names):
        dict_ret = datasets_utils.Dictionary_spec()

        print(f'Dictionary constructed, len = {len(dict_ret)}, (in transformer based mode, nvocab became useless).')
        return dict_ret
    
    def build_dict_index(self, file_names):
        counter = Counter()
        for file_name in file_names:
            if file_name is None: continue
            assert os.path.exists(file_name), 'File [{}] doesn\'t exists.'.format(file_name)

            with open(file_name, 'r', encoding = self.encoding) as f:
                lines = tqdm(f.readlines(), ascii = True)
                for line in lines:
                    lines.set_description(f"Building dict from {file_name.split('/')[-1]}")
                    if ': null,' in line: line = re.sub(': null,', ': \"null\",', line)
                    dt = json.loads(line)
                
                    counter.update(datasets_utils.tokenize(dt['text'], pre_eos = False, end_eos = False))
                    counter.update(datasets_utils.tokenize(dt['headline'], pre_eos = False, end_eos = False))

        dictionaty = datasets_utils.Dictionary()
        for wid, freq in counter.most_common(self.config.nvocab):
            if not wid in [dictionaty.pad, dictionaty.eos, dictionaty.eoe, dictionaty.unk]:
                dictionaty.add_word(wid, freq)
        print(f'dictionaty constructed, len = {len(dictionaty) - dictionaty.special} + {dictionaty.special} = {len(dictionaty)}')
        
        return dictionaty

    def build_dict_char(self, file_names):
        dict_ret = datasets_utils.build_char_dict()
        print(f'Dictionary constructed, len = {len(dict_ret)}')
        return dict_ret

    @classmethod
    def setup_dataset(cls):
        return cls























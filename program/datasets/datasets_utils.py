import os, sys, importlib, string, re, torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel

class Dictionary(object):
    def __init__(self, special = True):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

        self.pad = '<pad>'
        self.eos = '<eos>'
        self.eoe = '<eoe>'
        self.unk = '<unk>'

        self.add_word(self.pad)
        self.add_word(self.eos)
        self.add_word(self.eoe)
        self.add_word(self.unk)

        self.special = len(self.idx2word)

    def add_word(self, word, freq = 1): # 检查这个函数并从这里开始继续
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        token_id = self.word2idx[word]
        self.counter[word] += freq
        self.total += freq
        return token_id
    
    def get_index(self, word, unk = False):
        if word in self.word2idx:
            return self.word2idx[word]
        if unk:
            return self.word2idx[self.unk]
        raise RuntimeError('Unknown word [{}]'.format(word))

    def pop(self, key):
        del self.counter[key]
        self.idx2word.remove(key)
        self.word2idx.pop(key)

    def __len__(self):
        return len(self.idx2word)

class Dictionary_spec(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word, index):
        if word in self.word2idx:
            assert index == self.word2idx[word], f'Word index cannot be changed, got {word} changes from {self.word2idx[word]} to {index}.'
        else:
            self.word2idx[word] = index
            self.idx2word[index] = word
        input(word)
        input(index)
        input(self.word2idx[word])
    
    def add_line(self, words, indices):
        for word, index in zip(words, indices[0]):
            self.add_word(word, index)

    def __len__(self):
        return len(self.idx2word)

class transformer_tokenizer(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(mode)
    def __call__(self, line):
        ret = self.tokenizer(line, return_tensors = "pt")
        ret = ret['input_ids'].squeeze()
        ret = ret[: 512]
        return ret

class transformer_model(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained) 
    
    def forward(self, vec, ret_mode = 0):
        ''' ret_mode = 0: only return embedding,
            ret_mode = 1: only return vector
            ret_mode = 2: return embedding and vector'''
        vec = {'input_ids': vec}
        ret = self.model(**vec)

        if ret_mode == 0:
            return ret[0]
        elif ret_mode == 1:
            return ret[1]
        else:
            return ret

def build_char_dict():
    ret = Dictionary()
    charbox = list('abcdefghijklmnopqrstuvwxyz 0123456789!@#$%^&*()_+-=[]\\;\',./:"<>?|}{~`')
    for c in charbox:
        ret.add_word(c)
    return ret

def remove_punctuations(text):
    b = ''.join(c for c in text if c not in string.punctuation)
    return b

def tokenize(line, spliter = ' ', pre_eos = False, end_eos = True, lower = True, remove_punc = False, character = False):
    if lower: line = line.lower()
    if remove_punc: line = remove_punctuations(line)
    line = re.compile("\s+").sub(" ", line) # remove extra spaces ('a  b  c' -> 'a b c')
    line = line.strip()

    words = list(line) if character else line.strip().split(spliter)

    if pre_eos: words = ['<eos>'] + words
    if end_eos: words.append('<eoe>')

    return words

def pad_sequence_index(vecs, pad_mode = 'post', pad_value = 0):
    max_len = max([len(tmp) for tmp in vecs])
    ret = torch.ones([len(vecs), max_len], dtype = torch.int32) * pad_value

    for idx, vec in enumerate(vecs):
        t_vec = vec #torch.from_numpy(vec.astype(np.int32))
        if pad_mode == 'post':
            ret[idx][:len(vec)] = t_vec
        else:
            ret[idx][-len(vec):] = t_vec
    return ret.contiguous()

def pad_sequence_spec(vecs, pad_mode = 'post', pad_value = 0):
    max_len = max([tmp.shape[-1] for tmp in vecs])
    ret = torch.ones([len(vecs), max_len], dtype = torch.int32) * pad_value
    for idx, vec in enumerate(vecs):
        t_vec = vec[0] #torch.from_numpy(vec.astype(np.int32))
        if pad_mode == 'post':
            ret[idx][:len(t_vec)] = t_vec
        else:
            ret[idx][-len(t_vec):] = t_vec
    return ret.contiguous()

def pad_sequence(samples, pad_mode = 'post', pad_value = 0):
    sources = []
    wizards = []
    targets = []
    for sample in samples:
        sources.append(sample[0])
        wizards.append(sample[1])
        targets.append(sample[2])
    
    max_len = max([len(tmp) if tmp is not None else 0 for tmp in sources])
    ret = torch.ones([len(sources), max_len], dtype = torch.int32) * pad_value

    for idx, vec in enumerate(sources):
        t_vec = vec
        if pad_mode == 'post':
            ret[idx][:len(t_vec)] = t_vec
        else:
            ret[idx][-len(t_vec):] = t_vec
    return ret.contiguous(), torch.tensor(wizards), torch.tensor(targets)

def get_embedding_matrix(embed_dim, dictionary, pretrained):
    pretrained = "datasets/glove.6B/" + pretrained + '.txt'
    missing = 0
    glove_dict = {}
    emb_weight = torch.FloatTensor(len(dictionary), embed_dim)
    torch.nn.init.xavier_uniform_(emb_weight)
    with open(pretrained, 'r', encoding = 'utf-8') as f:
        lines = tqdm(f.readlines(), ascii = True)
        for line in lines:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            glove_dict[word] = embedding
            # lines.set_description(f"Loaing pretrained: {word.ljust(16,' ')}")  
            lines.set_description(f"Loaing pretrained...")  
            
    for idx, (key, v) in enumerate(dictionary.word2idx.items()):
        if key in glove_dict:
            emb_weight[idx] = torch.tensor(glove_dict[key])
        else:
            missing += 1
    
    emb_weight = emb_weight.numpy()
    print(f'Dictionary has been built, missing token = [{missing}]')
    return emb_weight

def get_embedding_char(word_count, embed_dim):
    emb_weight = np.zeros([word_count, embed_dim], dtype = np.float32)
    for i in range(word_count):
        emb_weight[i, i] = 1.0
    return emb_weight

def shuffle(source, wizard, target):
    indices = np.arange(len(target))
    np.random.shuffle(indices)
    source = source[indices]
    if wizard is not None and len(wizard) > 0: 
        wizard = wizard[indices]
    else:
        wizard = None
    if target is not None and len(target) > 0: 
        target = target[indices]
    else:
        target = None

    return source, wizard, target






















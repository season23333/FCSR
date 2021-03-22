import os, sys, time, random, torch, functools, math, json, hashlib, warnings, nni, logging, argparse
import numpy as np
import torch.nn as nn
from program import datasets, models, optimizers, utils, modules, criterions
from tensorboardX import SummaryWriter

try:
    import apex
    from apex import amp
    print('NVIDIA apex is available.')
except:
    warnings.warn("apex is unavailable, failed to import.")

class hyper_params():
    def __init__(self, args):
        self.cuda = 2
        self.clip = 5.0 # 梯度裁剪
        self.embed_dim = 300 # embed 维数
        self.hidden_dim = 300 # 模型维数
        self.log_interval = 100 # 输出日志间隔
        self.lr = 0.003#5232800179401513 # 初始学习率，所有学习率衰减函数都根据初始学习率计算
        self.lr_update = True # 学习率是否更新
        self.max_epoch = 1 # 最大训练轮数
        self.optimizer = 'adam' # 优化器
        self.template_count = 10 # template 第2维，第1维固定为1
        self.tensorlog_path = None # tensorlog存储位置
        self.warmup = 0 # 学习率预热
        self.weight_decay = 0.00000 #权重衰减
        self.nni = False
        self.seed = 1

        self.batch_size = 64
        self.dataset_name = 'cagnews'
        self.pretrained = 'glove.6B.300d'
        self.embedding_trainable = False
        self.entrance = 'freestyle'
        self.nvocab = 999999999
        self.nvocab_src = 999999999
        self.nvocab_tgt = 999999999
        self.force = False

        self.model_name = 'fcsr_opt'
        self.nlayer = 1 # 层数
        self.nhead = 1 # 头数
        self.dropp = 0.5#235441558604015 # dropout

        if self.seed > 0: 
            seed = self.seed
            seed_torch(seed)
        else:
            seed = random.randint(1, 999999999)
            seed_torch(seed)

        for k, v in args.items():
            if hasattr(self, k) and v is not None: setattr(self, k, v)

class freestyle():
    def __init__(self, hp):
        self.hp = hp
        self.writer = SummaryWriter(hp.tensorlog_path) if hp.tensorlog_path is not None else None
        self.time_start = time.time()

        self.dataset = utils.load_dataset(self.hp)
        self.hp.num_label = self.dataset.num_label
        self.parameter_template = None
        self.embedding = modules.Embedding(ntoken = len(self.dataset.dict), 
                                           embed_dim = hp.embed_dim, 
                                           pretrained_matrix = self.dataset.embedding_matrix, 
                                           trainable = hp.embedding_trainable,
                                          )
        self.model1 = mfpnet1(hp, self.dataset.num_label)
        self.model2 = mfpnet2(hp, self.dataset.num_label)

        # self.model1 = nn.DataParallel(self.model1, device_ids = [0, 1]).cuda()
        # self.model2 = nn.DataParallel(self.model2, device_ids = [0, 1]).cuda()
        # self.embedding = nn.DataParallel(self.embedding, device_ids = [0, 1]).cuda()

        self.model1 = self.model1.to('cuda:0')
        self.model2 = self.model2.to('cuda:0')
        self.embedding = self.embedding.to('cuda:0')

        self.all_params = list(self.model1.parameters()) + list(self.model2.parameters()) + list(self.embedding.parameters())
        self.optimizer = optimizers.load_optimizer(hp.optimizer)()(params = self.all_params, optimizer_config = self.hp)

        self.stack1 = result_stack()
        self.stack2 = result_stack()
        self.ret_stack1 = result_stack()
        self.ret_stack2 = result_stack()
        self.recoder = recoder()
                  
    def __call__(self):
        for cur_epoch in range(self.hp.max_epoch):
            #  --- train
            if self.hp.lr_update:
                utils.adjust_learning_rate(self.optimizer, cur_epoch, self.hp.max_epoch, self.hp.lr)
            else:
                utils.set_learning_rate(self.optimizer, self.hp.lr, False)
            _ = self.train_epoch(cur_epoch, self.dataset.config.dataset_name, self.dataset.loader['trn'], 'trn')
            # --- val and test
            val_dict1, val_dict2 = self.train_epoch(cur_epoch, self.dataset.config.dataset_name, self.dataset.loader['val'], 'val')
            tst_dict1, tst_dict2 = self.train_epoch(cur_epoch, self.dataset.config.dataset_name, self.dataset.loader['tst'], 'tst')
            acc_tmp = (tst_dict2['correct'] / tst_dict2['total'])
            if self.hp.nni: 
                nni.report_intermediate_result(acc_tmp)
            self.recoder.push(cur_epoch, val_dict2, tst_dict2)
            self.recoder.print()
                
        fin_epoch, fin_eval = self.recoder.best_val_epoch, self.recoder.fin_tst
        if self.hp.nni:
            nni.report_final_result(fin_eval)
        print(f'[{self.hp.max_epoch}] epoches complete, output results = {fin_eval} at epoch [{fin_epoch}], seed = {self.hp.seed}.')
        self.evaluate()
        
    def train_epoch(self, epoch_idx, dataset_name, batch_loader, tvt):
        assert tvt in ['trn', 'val', 'tst'], 'tvt must be chosen from [trn, val, tst].'
        if tvt == 'trn': 
            self.embedding.train()
            self.model1.train()
            self.model2.train()
        else: 
            self.embedding.eval()
            self.model1.eval()
            self.model2.eval()

        backward_loss = []
        extra_input = None

        for batch_idx, (src, wiz, tgt) in enumerate(batch_loader):
            # warm up
            if epoch_idx == 0 and self.hp.warmup > 0 and batch_idx < self.hp.warmup:
                utils.set_learning_rate(self.optimizer, self.hp.lr * cnt / self.hp.warmup, False)

            cur_batch = [
                    src.to('cuda:0') if src is not None and torch.cuda.is_available() else src,
                    wiz.to('cuda:0') if src is not None and torch.cuda.is_available() else wiz,
                    tgt.to('cuda:0') if src is not None and torch.cuda.is_available() else tgt,
                ]
            # cur_batch = [
            #         src.cuda() if src is not None and torch.cuda.is_available() else src,
            #         wiz.cuda() if src is not None and torch.cuda.is_available() else wiz,
            #         tgt.cuda() if src is not None and torch.cuda.is_available() else tgt,
            #     ]
            # forward
            ret_dict1, rep = self.model1(cur_batch, self.embedding, extra_input, None, self.writer)
            self.stack1.push_dict(ret_dict1)
            self.ret_stack1.push_dict(ret_dict1)
            ret_dict2 = self.model2(cur_batch, self.embedding, rep, None, self.writer)
            self.stack2.push_dict(ret_dict2)
            self.ret_stack2.push_dict(ret_dict2)
            if tvt == 'trn': 
                loss1 = functools.reduce(lambda x, y: x + y, ret_dict1['loss'])
                if loss1.isnan():
                    warnings.warn(f'loss1 = {loss1}.')
                loss1.backward(retain_graph=True)
                # 有选择的权重衰减策略
                # if self.hp.weight_decay > 0: self.task.apply(self.apply_weight_decay)
                # if self.hp.clip > 0: torch.nn.utils.clip_grad_norm_(self.all_params, self.hp.clip)
                # self.optimizer.step()
                # self.optimizer.zero_grad()

                
                loss2 = functools.reduce(lambda x, y: x + y, ret_dict2['loss'])
                if loss2.isnan():
                    warnings.warn(f'loss2 = {loss2}.')
                loss2.backward()
                # 有选择的权重衰减策略
                if self.hp.weight_decay > 0: 
                    self.model1.apply(self.apply_weight_decay)
                    self.model2.apply(self.apply_weight_decay)
                if self.hp.clip > 0: torch.nn.utils.clip_grad_norm_(self.all_params, self.hp.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # eval stage, omit loss     
            else:
                ret_dict1['loss'] = 0
                ret_dict2 = None
            
            # --- push results

            # log results
            log_interval = self.hp.log_interval if tvt == 'trn' else len(batch_loader)
            if (batch_idx + 1) % log_interval == 0 or batch_idx == (len(batch_loader) - 1):
                sp1 = self.stack1.pop_dict()
                sp2 = self.stack2.pop_dict()
                self.print_batches(epoch_idx, dataset_name, sp1, batch_idx + 1, len(batch_loader), tvt)
                self.print_batches(epoch_idx, dataset_name, sp2, batch_idx + 1, len(batch_loader), tvt)
        
        return self.ret_stack1.pop_dict(), self.ret_stack2.pop_dict()

    def print_batches(self, epoch_idx, dataset_name, task_ret_set, batch_cnt, batch_num, tvt):
        loss_detach = task_ret_set['loss_detach']#.cpu().numpy()
        correct = task_ret_set['correct']#.cpu().numpy()
        total = task_ret_set['total']#.cpu().numpy()
        acc = correct / total
        bpc = loss_detach / math.log(2)
        
        time_now = time.time()
        time_span = time_now - self.time_start
        self.time_start = time_now
        tmp = '|Epoch {}|{}->{}|lr {:2.2e}|batch {}/{}|loss {:4.3f}|bpc {:4.3f}|acc {:4.2f}|correct {:5d}|total {:5d}|{:3.2f}s|'.format(
            str(epoch_idx).rjust(2), self.dataset.task_type, dataset_name.rjust(13), self.optimizer.param_groups[0]['lr'], str(batch_cnt).rjust(4), 
            str(batch_num).rjust(4), loss_detach, bpc, acc * 100, int(correct), int(total), time_span)
        print(tmp)

    def apply_weight_decay(self, m):
        if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d)) and m.weight.grad is not None:
            m.weight.grad += m.weight * self.hp.weight_decay

    def evaluate(self, dataset = None):
        if dataset is None:
            dataset = self.dataset
        
        tmp = time.localtime(time.time())
        file_name = f'saved_logs/{dataset.config.dataset_name}_{tmp[1]}_{tmp[2]}_{tmp[3]}_{tmp[4]}.txt'
        logits = []
        targets = []
        for (src, wiz, tgt) in dataset.loader['tst']:
            cur_batch = [
                src.to('cuda:0') if src is not None and torch.cuda.is_available() else src,
                wiz.to('cuda:0') if src is not None and torch.cuda.is_available() else wiz,
                tgt.to('cuda:0') if src is not None and torch.cuda.is_available() else tgt,
            ]
            ret_dict1, rep = self.model1(cur_batch, self.embedding, None, None, self.writer)
            ret_dict2 = self.model2(cur_batch, self.embedding, rep, None, self.writer)

            logits.append(ret_dict2['rep'].cpu())
            targets.append(tgt.cpu())
        logits = torch.cat(logits, dim = 0)
        targets = torch.cat(targets).unsqueeze(1)
        forsave = torch.cat([targets, logits], dim = 1).numpy()
        np.savetxt(file_name, forsave)
        utils.plot_roc(file_name)

def seed_torch(seed = 1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
    print(f'Random seed was set with {seed}.')

class recoder():
    def __init__(self):
        self.max_val = -1
        self.max_tst = -1
        self.best_val_epoch = -1
        self.bect_tst_epoch = -1
        self.fin_tst = -1
    
    def push(self, cur_epoch, dict_val, dict_tst):
        val_acc = dict_val['correct'] / dict_val['total']
        tst_acc = dict_tst['correct'] / dict_tst['total']

        if tst_acc > self.max_tst:
            print(f'New best tst acc [{tst_acc}] > previous tst acc [{self.max_tst}] detected at epoch [{cur_epoch}]')
            self.max_tst = tst_acc
            self.bect_tst_epoch = cur_epoch
        
        if val_acc > self.max_val:
            print(f'New best val acc [{val_acc}] > previous val acc [{self.max_val}] detected at epoch [{cur_epoch}]')
            self.max_val = val_acc
            self.best_val_epoch = cur_epoch
            self.fin_tst = tst_acc
            print(f'New output acc was set with {tst_acc} at epoch {cur_epoch}')

    def print(self):
        tmp1 = f'Max val acc = {self.max_val}, output tst acc = {self.fin_tst} at epoch {self.best_val_epoch}'
        tmp2 = f'Max tst_acc = {self.max_tst} at epoch {self.bect_tst_epoch}'
        print(tmp1)
        print(tmp2)

class result_stack():
    def __init__(self):
        self.data = None
    
    def push_dict(self, result):
        if self.data is None:
            self.data = {}
        for k, v in result.items():
            if k == 'loss' or isinstance(v, str) or k == 'rep': continue
            if not k in self.data:
                self.data[k] = []
            self.data[k].append(v)
    
    def pop_dict(self):
        if self.data is None: return None
        dict_ret = {}
        for k, v in self.data.items():
            if k == 'correct' or k == 'total':
                dict_ret[k] = torch.sum(torch.tensor(v)).cpu().numpy()
            else:
                dict_ret[k] = torch.mean(torch.tensor(v)).cpu().numpy()
        self.data = None
        return dict_ret

class Trans(nn.Module):
    def forward(self, vec):
        return vec.transpose(1, 2)

class mfpnet2(nn.Module):
    def __init__(self, hp, num_label):
        super().__init__()
        self.embed_dim = hp.embed_dim
        self.hidden_dim = hp.hidden_dim
        self.num_label = num_label

        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        self.pm = modules.PositionalEmbedding(self.hidden_dim)
        self.net = models.load_model(hp.model_name)(hp, None)
        self.fnn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.criterion = criterions.load_criterion('ced_spec')(self.hidden_dim, self.num_label)
        self.dropi = nn.Dropout(0.5)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = embedding(source.long())
        if self.transfer: vec = self.transfer(vec)
        # extra_output = {'to_loss': [vec, target]}

        # vec = self.pm(vec) # positional embedding
        vec, _ = self.net([vec, None, None], None, None, None, None)
        vec = self.fnn(vec)
        vec = torch.relu(vec)

        d = self.nb_alg(vec, extra_input)
        vec = vec - self.nb_alg(vec, d).reshape(vec.shape)

        vec = self.linear(vec)
        vec = torch.relu(vec)
        vec = self.dropi(vec)
        # vec = torch.mean(vec, dim = 1)
        ret_dict = self.criterion(vec, target)
        return ret_dict
    
    def nb_alg(self, u, v):
        shape = u.shape
        u = u.reshape([shape[0], -1])
        v = v.reshape([shape[0], -1])
        mag = (torch.sum(v ** 2, axis = -1) ** 0.5).reshape([shape[0],1])
        normalized = (1.0 / mag)  * v
        projection = u * u * normalized

        return projection

class mfpnet1(nn.Module):
    def __init__(self, hp, num_label):
        super().__init__()
        self.embed_dim = hp.embed_dim
        self.hidden_dim = hp.hidden_dim
        self.num_label = num_label

        self.transfer = nn.Linear(self.embed_dim, self.hidden_dim) if not self.embed_dim == self.hidden_dim else None
        self.pm = modules.PositionalEmbedding(self.hidden_dim)
        self.net = models.load_model(hp.model_name)(hp, None)
        self.fnn = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropi = nn.Dropout(0.5)
        self.criterion = criterions.load_criterion('ced_spec')(self.hidden_dim, self.num_label)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        vec = embedding(source.long())
        if self.transfer: vec = self.transfer(vec)
        # extra_output = {'to_loss': [vec, target]}

        # vec = self.pm(vec) # positional embedding
        vec, _ = self.net([vec, None, None], None, None, None, None)
        vec = self.fnn(vec)
        rep = torch.relu(vec)

        vec = FlipGradientBuilder.apply(rep)
        vec = self.linear(vec)
        vec = torch.relu(vec)
        vec = self.dropi(vec)
        # vec = torch.mean(vec, dim = 1)
        ret_dict = self.criterion(vec, target)
        return ret_dict, rep

def load_params():
    parser = argparse.ArgumentParser()
    pa = parser.add_argument
    #---str
    pa("--lr", type = float, default = None, help = "lr")
    pa("--lr_update", type = bool, default = None, help = "lr")
    pa("--weight_decay", type = float, default = None, help = "weight_decay")
    pa("--nni", type = bool, default = None, help = "nni")
    pa("--batch_size", type = int, default = None, help = "batch_size")
    pa("--embedding_trainable", type = bool, default = None, help = "embedding_trainable")
    pa("--nvocab", type = int, default = None, help = "nvocab")
    pa("--dropp", type = float, default = None, help = "dropp")

    args = parser.parse_args()
    return args

class FlipGradientBuilder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu    
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input *= -torch.log(grad_input + 1)
        # grad_input *= (1.0 / (grad_input + 1e-12))
        return grad_input

if __name__ == '__main__':
    args = load_params().__dict__
    hp = hyper_params(args)
    print(f'hyper_param = {hp.__dict__.items()}')
    freestyle(hp)()



























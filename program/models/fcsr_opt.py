import torch
from . import basic_model, register_model
from .. import modules
import torch.nn as nn
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, initializer = nn.init.xavier_uniform_):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
        initializer(self.net.weight)
        nn.init.constant_(self.net.bias, 1.0)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, vec):
        
        vec = self.net(vec)
        vec = self.norm(vec)
        vec = torch.relu(vec)
        return vec


@register_model('fcsr_opt')
class fcsr_opt(basic_model):
    def __init__(self, model_config, template):
        super().__init__(
            'fcsr', 
            embed_dim = model_config.embed_dim, 
            hidden_dim = model_config.hidden_dim, 
        )
        self.makeup = self_attn(self.hidden_dim, self.hidden_dim, self.hidden_dim)

        self.gcfv = Linear(self.hidden_dim, self.hidden_dim)
        self.gcfc = Linear(2 * self.hidden_dim, self.hidden_dim)

        self.gcdv = Linear(self.hidden_dim, self.hidden_dim)
        self.gcdc = Linear(2 * self.hidden_dim, self.hidden_dim)

        self.gciv = Linear(self.hidden_dim, self.hidden_dim)
        self.gcic = Linear(2 * self.hidden_dim, self.hidden_dim)

        self.gcov = Linear(self.hidden_dim, self.hidden_dim)
        self.gcoc = Linear(2 * self.hidden_dim, self.hidden_dim)

        self.gdfv = Linear(self.hidden_dim, self.hidden_dim)
        self.gdfc = Linear(self.hidden_dim, self.hidden_dim)
        self.gdfh = Linear(self.hidden_dim, self.hidden_dim)

        self.gddv = Linear(self.hidden_dim, self.hidden_dim)
        self.gddc = Linear(self.hidden_dim, self.hidden_dim)
        self.gddh = Linear(self.hidden_dim, self.hidden_dim)

        self.gdiv = Linear(self.hidden_dim, self.hidden_dim)
        self.gdic = Linear(self.hidden_dim, self.hidden_dim)
        self.gdih = Linear(self.hidden_dim, self.hidden_dim)

        self.gdov = Linear(self.hidden_dim, self.hidden_dim)
        self.gdoc = Linear(self.hidden_dim, self.hidden_dim)
        self.gdoh = Linear(self.hidden_dim, self.hidden_dim)

        self.drop = nn.Dropout(model_config.dropp)
        self.lnv = nn.LayerNorm(self.hidden_dim)
        self.lnc = nn.LayerNorm(self.hidden_dim * 2)

    def cite_gate(self):
        return torch.relu

    def cite_sele(self):
        return torch.tanh

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        
        vec = modules.prepare_input(batch, embedding)
        extra_output = {'to_loss': [vec, target]}

        fvb, fva = self.context(vec, 999999, )
        cov = torch.cat([fvb, fva], dim = 2)

        # hasnan1 = vec.is_nan().any() or cov.is_nan().any()
        vec = self.lnv(vec)
        cov = self.lnc(cov)
        # hasnan2 = vec.is_nan().any() or cov.is_nan().any()
        # if not hasnan1 == hasnan2:
        #     input('batchnorm de guo')

        gcf = self.cite_gate()(self.gcfv(vec) + self.gcfc(cov))
        gcd = self.cite_gate()(self.gcdv(vec) + self.gcdc(cov))
        gci = self.cite_gate()(self.gciv(vec) + self.gcic(cov))
        gco = self.cite_gate()(self.gcov(vec) + self.gcoc(cov))

        cc = gcf * vec + gcd * fvb + gci * fva
        ch = gco * self.cite_sele()(cc)

        gdf = self.cite_gate()(self.gdfv(vec) + self.gdfc(cc) + self.gdfh(ch))
        gdd = self.cite_gate()(self.gddv(vec) + self.gddc(cc) + self.gddh(ch))
        gdi = self.cite_gate()(self.gdiv(vec) + self.gdic(cc) + self.gdih(ch))
        gdo = self.cite_gate()(self.gdov(vec) + self.gdoc(cc) + self.gdoh(ch))

        dc = gdf * vec + gdd * cc + gdi * ch
        dh = gdo * self.cite_sele()(dc)

        ret = torch.cat([cc, ch, dc, ch], dim = 1)
        ret = self.drop(ret)

        ret = ret / ret.shape[-1] ** 0.5
        # ret = torch.mean(ret, dim = 1)
        ret = self.makeup(ret)
        return ret, extra_output
    
    def context(self, vec, window, alpha = 0.2):
        window = min(vec.shape[1] - 1, window)
        bef = torch.zeros(vec.shape, device = vec.device, dtype = vec.dtype)
        aft = torch.zeros(vec.shape, device = vec.device, dtype = vec.dtype)
        # bef = vec.clone().detach()
        # aft = vec.clone().detach()
        for idx in range(window):
            vec = vec * alpha
            pad = torch.zeros(vec.shape[0], idx  + 1, vec.shape[2], device = vec.device, dtype = vec.dtype)
            lft_pad = torch.cat([pad, vec], dim = 1)[:, :vec.shape[1], :]
            bef += lft_pad
            rit_pad = torch.cat([vec, pad], dim = 1)[:, -vec.shape[1]:, :]
            aft += rit_pad
        return bef, aft
    
    def Bef_aft_n(self, len, alpha = 0.2, window = 1):
        bef = np.zeros([len,len],dtype = np.float32)
        bef = np.tril(bef, 0)

        for i in range(len):
            factor = 1.0
            bound = min(i + window + 1, len)
            for j in range(i, bound):
                bef[j, i] = factor
                factor *= alpha
        for i in range(len):
            bef[i,i] = 0.0
        aft = bef.copy().T
        return torch.tensor(bef), torch.tensor(aft)

    @classmethod
    def setup_model(cls):
        return cls

class self_attn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.wq = nn.Linear(in_dim, hidden_dim)
        self.wk = nn.Linear(in_dim, hidden_dim)
        self.nq = nn.LayerNorm(in_dim)
        self.nk = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, vec):
        q = self.wq(vec)
        k = self.wq(vec)

        ret = torch.einsum('bsx,bsy->bxy', q, k)
        ret = torch.relu(ret)
        # input(ret.shape)
        ret = self.linear(ret)
        ret = ret.reshape([vec.shape[0], -1])
        return ret

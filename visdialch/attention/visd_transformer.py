import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pdb
import numpy as np 

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    #real layer-norm
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

    #apply L2-normalization
    '''
    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        return x
    '''
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config['n_head'] == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer("spot", torch.eye(n_ctx, n_ctx).view(1, 1, n_ctx, n_ctx))
        self.n_head = config['n_head']
        self.split_size = n_state
        self.scale = scale
        self.attn_dropout = nn.Dropout(config['attn_pdrop'])
        self.resid_dropout = nn.Dropout(config['resid_pdrop'])
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        """
        q (tensor): query tensor, (batch, head, seq_length, head_features)
        k (tensor): key tensor, (batch, head, head_features, seq_length)
        v (tensor): value tensor, (batch, head, seq_length, head_features)
        """
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        # Here the bias b also serves as the mask to remove future information
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def _attn_dual(self, kx, vx, qy, ky, vy):
        """
        This attention aims to extract residual information from x associated to each item in y.
        Specific in visual dialog case, x and y have the same sequence length(x, y = history, question).
        kx (tensor): key tensor of x, (batch, head, head_features, seq_length)
        vx (tensor): value tensor of x, (batch, head, seq_length, head_features)
        qy (tensor): query tensor of y, (batch, head, seq_length, head_features)
        ky (tensor): key tensor of y, (batch, head, head_features, seq_length)
        vy (tensor): value tensor of y, (batch, head, seq_length, head_features)
        """
        #y to x attention weight
        #pdb.set_trace()
        y2x_w = torch.matmul(qy, kx) #(batch, head, seq_length, seq_length), compute attention weight of y respect to x
        if self.scale:
            y2x_w = y2x_w / math.sqrt(vx.size(-1))
        seq_len = y2x_w.size(-1)
        byx = self.bias[:, :, 0:seq_len, 0:seq_len] #(1, 1, seq_len, seq_len)
        y2x_w = y2x_w*byx - 1e10*(1 - byx) #(batch, head, seq_len, seq_len)
        
        #y's self weight to enable regression
        y2y_w = torch.matmul(qy, ky) #(batch, head, seq_length, seq_length), compute y's self-attention weight 
        if self.scale:
            y2y_w = y2y_w / math.sqrt(vy.size(-1))
        byy = self.spot[:, :, 0:seq_len, 0:seq_len] #(1, 1, seq_len, seq_length)
        y2y_w = y2y_w*byy - 1e10*(1 - byy) #(batch, head, seq_length, seq_length)

        w = torch.cat((y2x_w, y2y_w), dim=-1) #(batch, head, seq_length, 2*seq_length)
        v = torch.cat((vx, vy), dim=-2) #(batch, head, 2*seq_length, head_features)
        w = nn.Softmax(dim=-1)(w) #(batch, head, seq_length, 2*seq_length)
        w = self.attn_dropout(w)
        y_att_x = torch.matmul(w, v) #(batch, head, seq_length, head_features)
        return y_att_x, w

    def merge_heads(self, x):
        """
        x (tensor): attention results, (batch, head, seq_length, head_features)
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # (batch, seq_length, head*head_features), in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        """
        x (tensor): (batch, seq_length, n_state)
        """
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, y = None):
        """
        x (tensor): input tensor, (batch, seq_length, nx)
        y (tensor): optional input tensor, None or (batch, seq_length, ny), if y is not None, y should be treated as masked query while x is the masked value
        """
        x = self.c_attn(x) #(batch, seq_length, n_state*3)
        qx, kx, vx = x.split(self.split_size, dim=2) #(batch, seq_length, n_state)
        qx = self.split_heads(qx) #(batch, head, seq_length, head_features)
        kx = self.split_heads(kx, k=True) #(batch, head, head_features, seq_length)
        vx = self.split_heads(vx) #(batch, head, seq_length, head_features)
        if y is None:
          a = self._attn(qx, kx, vx)
        else:
          y = self.c_attn(y)
          qy, ky, vy = y.split(self.split_size, dim=2)
          qy = self.split_heads(qy)
          ky = self.split_heads(ky, k=True)
          vy = self.split_heads(vy)
          a, w = self._attn_dual(kx, vx, qy, ky, vy)
        a = self.merge_heads(a)
        a = self.c_proj(a) #(batch, seq_length, nx)
        a = self.resid_dropout(a)
        if y is None:
          return a
        else:
          # a: #(batch, head, seq_length, head_features)
          # w: #(batch, head, seq_length, 2*seq_length)
          return a, w
        


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config['n_embd']
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config['resid_pdrop'])

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2)
        return h2


class Elem(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Elem, self).__init__()
        n_embd = config['n_embd']
        self.ln_1 = LayerNorm(n_embd, eps=config['layer_norm_epsilon'])
        self.attn = Attention(n_embd, n_ctx, config, scale)
        self.ln_2 = LayerNorm(n_embd, eps=config['layer_norm_epsilon'])
        self.mlp = MLP(4 * n_embd, config)

    def forward(self, x):
        a = self.attn(self.ln_1(x))
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

class DualElem(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(DualElem, self).__init__()
        n_embd = config['n_embd']
        self.ln_1_x = LayerNorm(n_embd, eps=config['layer_norm_epsilon'])
        self.ln_1_y = LayerNorm(n_embd, eps=config['layer_norm_epsilon'])
        self.attn = Attention(n_embd, n_ctx, config, scale)
        self.ln_2_y = LayerNorm(n_embd, eps=config['layer_norm_epsilon'])
        self.mlp = MLP(4*n_embd, config)

    def forward(self, x, y):
        x = self.ln_1_x(x)
        y = self.ln_1_y(y)
        dy, w = self.attn(x, y)
        y = y + dy
        dy = self.mlp(self.ln_2_y(y))
        y = y + dy
        return y, w

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        self.encoder_x = Elem(n_ctx, config, scale)
        self.encoder_y = Elem(n_ctx, config, scale)
        self.encoder_y2x = DualElem(n_ctx, config, scale)

    def forward(self, x, y):
        x = self.encoder_x(x) #x self-attention
        #y = self.encoder_y(y) #y self-attention
        y, w = self.encoder_y2x(x, y) #compute x awared y
        return x, y, w

def positionEmbed(model_dim, pos_num, scale=10000):
    exp = -torch.FloatTensor([i//2*2 for i in range(model_dim)]).view(-1, 1)/model_dim
    w = torch.pow(scale, exp) #(model_dim, 1)
    pos = torch.FloatTensor([i for i in range(pos_num)]).view(1, -1) #(1, pos_nums)
    x = w * pos #(model_dim, pos_num)
    sin_mask = torch.zeros(model_dim, pos_num) #(model_dim, pos_num)
    ind = torch.tensor([i for i in range(model_dim) if i%2 == 0])
    sin_mask[ind] = 1 #(model_dim, pos_num)
    cos_mask = 1 - sin_mask #(model_dim, pos_num)
    pos_embed = torch.sin(x)*sin_mask + torch.cos(x)*cos_mask #(model_dim, pos_num)
    pos_embed = pos_embed.transpose(1,0) #(pos_num, model_dim)
    return pos_embed


class VisdTransformer(nn.Module):
    def __init__(self, config):
        super(VisdTransformer, self).__init__()
        self.n_layer = config['n_layer']
        self.n_embd = config['n_embd']

        self.pe = nn.Embedding.from_pretrained(positionEmbed(config['n_embd'], config['n_positions']))
        block = Block(config['n_ctx'], config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config['n_layer'])])
        self.ln_f_x = LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])
        self.ln_f_y = LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])

    def forward(self, x, y): #input_ids, position_ids=None, token_type_ids=None, past=None):
        """
        Args:
        x (tensor): (batch, seq_length, unified_hidden_dim)
        y (tensor): (batch, seq_length, unified_hidden_dim)
        Returns:
        x (tensor): masked x-x representation, (batch, seq_length, unified_hidden_dim)
        y (tensor): fusion of the masked y-y and y->x representations, (batch, seq_length, unified_hidden_dim)
                    Note: y->x representation is natually though as y grounded to x. 
        """
        batch, seq_len, _ = x.size()
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, seq_len)

        position_embeds = self.pe(position_ids) #(batch, seq_length, unified_hidden_dim)
        x_states = x + position_embeds
        y_states = y + position_embeds
        #presents = []
        y2xy_w = []
        for block in self.h:
            x_states, y_states, w = block(x_states, y_states)
            y2xy_w.append(w.clone().detach().cpu().numpy())
            #presents.append((x_states, y_states))
        x_states = self.ln_f_x(x_states)
        y_states = self.ln_f_y(y_states)
        # len(y2xy_w) == n_layer
        y2xy_w = np.array(y2xy_w)   # (n_layer, batch, head, seq_length, 2*seq_length)
        y2xy_w = np.swapaxes(y2xy_w, 0, 1)  # (batch, n_layer, head, seq_length, 2*seq_length)
        return x_states, y_states, y2xy_w

import math
import torch
from torch import nn
from thop import profile
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x    
    
    
class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaleDotProductAttention, self).__init__()
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)
    
    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x):
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        values, attention = self.scaled_dot_product(q, k, v)
        return values    
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        
        self.head_modules = nn.ModuleList()
        
        for _ in range(self.n_head):
            self.head_modules.append(ScaleDotProductAttention(d_model))
            
        self.w_concat = nn.Linear(d_model*n_head, d_model)

    def forward(self, x):
        
        out_tensor = []
        
        for head_mod in self.head_modules:
            out_tensor.append(head_mod(x)) 
            
        out = torch.concat(out_tensor, -1)
        out = self.w_concat(out)
        return out

    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(p=drop_prob)

        self.ffn       = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout2  = nn.Dropout(p=drop_prob)

    def forward(self, x):
        _x = x
        x = self.attention(x)
        
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        
        _x = x
        x = self.ffn(x)
      
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x
    
    
    
class Transformer_Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, 
                                            n_head=n_head, drop_prob=drop_prob))
        
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    
class Transformer_network(nn.Module):
    def __init__(self, input_dim, output_dim, dim, depth, heads, mlp_dim, pos_embed, 
                 dropout = 0., emb_dropout = 0.):
        super().__init__()
        
        self.fc         = nn.Linear(input_dim, dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(dim)) if pos_embed == 'True' else None
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_Encoder(d_model = dim, ffn_hidden = mlp_dim, n_head = heads,
                                               n_layers = depth, drop_prob = dropout)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_dim))

    def forward(self, x):
        x = self.fc(x)
        
        if self.pos_embedding != None:
            x += self.pos_embedding
        
        x = self.dropout(x)
        
        x = self.transformer(x)
        x = self.mlp_head(x)
        
        return x 
    
    def print_layer_param(self):
        print(self.pos_embedding[0])
    
    
if __name__ == "__main__":
    
    v = Transformer_network(input_dim = 964, output_dim = 1, dim = 512, depth = 2, heads = 2, mlp_dim = 512, 
                            dropout = 0.1, emb_dropout = 0.1, pos_embed = 'True')
    
    inp_tensor = torch.randn(1024, 964)
    label      = torch.randn(1024, 1)
    
    preds = v(inp_tensor) 
    
    print(label.size(), preds.size())
    
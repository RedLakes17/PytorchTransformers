import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0, 'd_model debe ser divisible entre num_heads'
        self.num_heads= num_heads
        self.d_model= d_model
        self.head_dim= d_model // num_heads

        self.query_linear= nn.Linear(d_model, d_model, bias=False) #Capas lineales
        self.key_linear= nn.Linear(d_model, d_model, bias=False)
        self.value_linear= nn.Linear(d_model, d_model, bias=False)

        self.output_linear= nn.Linear(d_model, d_model)
    
    def SplitHeads(self, x, batch_size): #Divide Q, K, V en cada head
        seq_lenght= x.size(1)
        x= x.reshape(batch_size, seq_lenght, self.num_heads, self.head_dim)
        return x.permute(0,2,1,3)
        
    def ComputeAttention(self, query, key, value, mask=None):
        scores= torch.matmul(query, key.transpose(-2,-1)) / (self.head_dim ** 0.5) #Producto punto entre Query y Key

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) #Nos aseguramos de que no halla mask

        attention_weights= F.softmax(scores, dim=-1) #Aplicamos activacion
        return torch.matmul(attention_weights, value) #Producto punto entre values y attention weights
    
    def CombineHeads(self, x , batch_size):
        x = x.permute(0,2,1,3).contiguous()
        return x.view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size= query.size(0)

        query= self.SplitHeads(self.query_linear(query), batch_size)
        key= self.SplitHeads(self.key_linear(key), batch_size)
        value= self.SplitHeads(self.value_linear(value), batch_size)

        attention_weights= self.ComputeAttention(query, key, value, mask)
        output= self.CombineHeads(attention_weights, batch_size)
        return self.output_linear(output)

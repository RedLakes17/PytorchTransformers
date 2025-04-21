#Asi se define un embedding
import torch
import math
import torch.nn as nn

#Embbeding
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int) -> None:
        super().__init__()
        self.vocab_size= vocab_size
        self.d_model= d_model
        self.embedding= nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) 
    
embbeding_layer1=InputEmbedding(vocab_size=10000, d_model=512) #Creamos un embedding
embbeded_output=embbeding_layer1(torch.tensor([[1,2,3,4],[5,6,7,8]]))
print(embbeded_output.shape)


#Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_lenght):
        super().__init__()

        pe= torch.zeros(max_seq_lenght, d_model)
        position= torch.arange(0, max_seq_lenght, dtype=torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

#Creamos un positional encoding
pos_encoding_layer = PositionalEncoding(d_model= 512, max_seq_lenght= 4)
pos_encoded_output= pos_encoding_layer(embbeded_output)
print(pos_encoded_output.shape)
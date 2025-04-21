#Veamos los parametros basicos que se le suministran a un transformer
import torch.nn as nn

model= nn.Transformer(
    d_model=512, #Dimension del embbeded sequence
    nhead=8, #Numero de attention heads
    num_encoder_layers=6, #Numero de capas encoder
    num_decoder_layers=6 #Numero de capas decoder
)

print(model)
import torch
import torch.nn as nn
import math

#Creating the input embedding layer of 512dim
class InputEmbeddings(nn.Module):
    def __init__(self,d_model: int, vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
#Creating the positional encoding of same size as input embedding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int,seq_len:int,dropout:float) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        #Creating a matrix of size (seq_len,d_model,1)
        pe=torch.zeros(seq_len,d_model)
        
        #Creating a vector of shape(seq_len)
        position=torch.arange(0,seq_len,dtype=float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model,2).float()*(-math.log(10000.0)/d_model))
        
        #Applying the sin to even position
        pe[:, 0::2]=torch.sin(position*div_term)
        #Applying the cos to odd position
        pe[:, 1::2]=torch.cos(position*div_term)


        pe=pe.unsqueeze(0)
        
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x=x+(self.pe[:, :x.shape[1], :]).requires_grad(False)
        return x.dropout(x)
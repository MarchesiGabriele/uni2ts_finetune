import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.r = 2 # TODO: make this a parameter in the config
        self.alpha = 16 # TODO: make this a parameter in the config
        self.scaling = self.alpha / self.r

        self.linear = linear
        self.linear.weight.requires_grad = False  # freeze original weight

        if self.linear.bias is not None: # if there is bias, freeze it
            self.linear.bias.requires_grad = False
        
        self.A = nn.Parameter(torch.randn(self.r, self.linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.linear.out_features, self.r))
        
        
    def forward(self, x):
        result = self.linear(x)  # include bias originale
        
        lora_update = (x @ self.A.T) @ self.B.T * self.scaling
        
        return result + lora_update

class LoRAEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.r = 2 # TODO: make this a parameter in the config
        self.alpha = 16 # TODO: make this a parameter in the config
        self.scaling = self.alpha / self.r

        self.embedding = embedding
        self.embedding.weight.requires_grad = False  # freeze original weight

        self.A = nn.Parameter(torch.randn(self.r, self.embedding.weight.shape[1]) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.embedding.weight.shape[1], self.r))
        
        
    def forward(self, x):
        result = self.embedding(x)  # include bias originale
        
        lora_update = (x @ self.A.T) @ self.B.T * self.scaling
        
        return result + lora_update



class LoRANorm(nn.Module):
    def __init__(self, W: nn.Parameter):
        super().__init__()
        k = W.shape[0]
        self.r = 2 # TODO: make this a parameter in the config
        self.alpha = 16 # TODO: make this a parameter in the config
        self.scaling = self.alpha / self.r

        # frozen parameter
        self.W = W
        self.W.requires_grad = False  # freeze original weight

        # LoRA parameters
        self.A = nn.Parameter(torch.randn(self.r, k) * 0.01) # 0.01 is for numerical stability
        self.B = nn.Parameter(torch.zeros(k, self.r))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_in)
        base = x * self.W  # moltiplicazione elemento per elemento con il peso originale
        lora = (x @ self.B) @ self.A * self.scaling  # moltiplicazioni matriciali per LoRA
        return base + lora


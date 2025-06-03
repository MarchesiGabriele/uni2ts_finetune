import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Original linear transformation
        result = self.linear(x)  # W @ x + b

        # LoRA adaptation: (B @ A) @ x = B @ (A @ x)
        lora_update = F.linear(F.linear(x, self.A), self.B) * self.scaling
        
        return result + lora_update

# class LoRAEmbedding(nn.Module):
#     def __init__(self, embedding: nn.Embedding):
#         super().__init__()
#         self.r = 2 # TODO: make this a parameter in the config
#         self.alpha = 16 # TODO: make this a parameter in the config
#         self.scaling = self.alpha / self.r

#         self.embedding = embedding
#         self.embedding.weight.requires_grad = False  # freeze original weight

#         self.A = nn.Parameter(torch.randn(self.r, self.embedding.weight.shape[0]) * 0.01)
#         self.B = nn.Parameter(torch.zeros(self.embedding.weight.shape[1], self.r))
        
#     @property
#     def weight(self):
#         print("A shape:", self.A.shape)
#         print("B shape:", self.B.shape) 
#         print("embedding shape:", self.embedding.weight.shape)
#         print("A @ B shape:", (self.A @ self.B).shape)
#         return self.embedding.weight + (self.A @ self.B) * self.scaling
        
#     def forward(self, x): # not used
#         pass



# class LoRANorm(nn.Module):
#     def __init__(self, W: nn.Parameter):
#         super().__init__()
#         k = W.shape[0]
#         self.r = 2 # TODO: make this a parameter in the config
#         self.alpha = 16 # TODO: make this a parameter in the config
#         self.scaling = self.alpha / self.r

#         # frozen parameter
#         self.W = W
#         self.W.requires_grad = False  # freeze original weight

#         # LoRA parameters
#         self.A = nn.Parameter(torch.randn(self.r, k) * 0.01) # 0.01 is for numerical stability
#         self.B = nn.Parameter(torch.zeros(k, self.r))

#     def forward(self, x):
#         base = x * self.W  
#         lora = (x @ self.B) @ self.A * self.scaling  
#         return base + lora


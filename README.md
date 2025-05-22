# Fine tuning
We want to fine tune the model on energy price time series, with the goal of predicting them. 
The model has not been pretrained on anny energy price TS. 

We know that energy price has seasonality, both daily and weekly. So we want the model to learn that. 
We know that there is a correlation between energy price and energy generation/load, we want the model to learn that.
The data has hourly frequency. The model should already be able to understand that. 


With lightweigth techniques we should also try to finetune the largest models (moirai base,large)


## Layers

### ProjectionIn and ProjectionOut 
The model has already been pretrained on billions of time series, so it already should know how to read one and how to convert the output into a mixture of distributions. 
For this reason those two layers should remain freezed. 

### Transformer Encoder Layer
which divides into
- SelfAttention Layer
- Feed Forward Layer
- Normalization Layer
- Dropout Layer
  - It is not impacted during training




### LayerNorm


## Techniques
### Full Fine Tune
[x] fft with stock hyperparams -> overfitting
[] fft with custom dropout=0.3, attndropout=0.3, weigth decay=1e-3, lr=1e-5, patience=5, -> TODO


### Only fine tune last layers of the transformers
[] Keep Projection layers frozen, only un-freeze the last transformers layers (selfatt, ffn, norm)
[] Keep Projection layers frozen, only un-freeze the last transformers layers (selfatt)
[] Keep Projection layers frozen, only un-freeze the last transformers layers (ffn)
[] Keep Projection layers frozen, only un-freeze the last transformers layers (norm)

### LoRA (Adapter)
[] Only on the self attention weigth matrices (q,k,v,out) (see lora paper)

### BitFit (fine tune only the bias)

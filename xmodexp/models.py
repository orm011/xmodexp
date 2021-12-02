import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import math
import transformers
from transformers import CLIPModel

class ContrastLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
    
    def loss(self, logits):
        target = torch.arange(logits.shape[0]).long().to(logits.device)
        return F.cross_entropy(logits, target, reduction='mean')
    
    def forward(self, vecs1, vecs2):
        assert vecs1.shape == vecs2.shape
        # cosine similarity as logits
        logits = (math.exp(self.tau) * vecs1) @ vecs2.t()
        return self.loss(logits) + self.loss(logits.t())

# see https://huggingface.co/transformers/_modules/transformers/models/clip/modeling_clip.html#CLIPModel

class CLIPModule(pl.LightningModule):
    def __init__(self, path_or_model_name, batch_size, tau):
        super().__init__()
        self.model = CLIPModel.from_pretrained(path_or_model_name)
        self.save_hyperparamters()
        
    def forward(self, inputs):
        return  self.model(**inputs)


    def basic_step(self, batch, train_or_val):
        batch_out = self.forward(batch)
        loss = self.loss_fn(batch_out['image'], batch_out['description'])
        self.log(f'{train_or_val}_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.basic_step(batch, 'val')
    
    def training_step(self, batch, batch_idx):
        return self.basic_step(batch, 'train')
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)
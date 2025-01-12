import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, 96, model_dim))
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=dropout
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(model_dim, output_dim)
    
    def forward(self, X):
        X = self.input_projection(X) 
        
        X = X + self.positional_encoding[:, :X.size(1), :]
        
        X = X.transpose(0, 1) 
        
        X = self.transformer_encoder(X)
        X = X[-1, :, :]
        
        y_pred = self.output_layer(X)
        
        return y_pred


## Importing packages and scripts

import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

### Lets create a input embedding layer class by subclassing torch.nn

class input_embedding_layer(torch.nn.Module):
    
    def __init__(self,input_shape,
                 output_shape,patch_size,
                 number_of_patches,
                 embedding_dropout):
        
        super().__init__()

        self.out_shape= output_shape
        self.patch_size= patch_size

        self.input_layer_stack = torch.nn.Sequential(
            
            torch.nn.Conv2d(in_channels=input_shape,
                            out_channels=output_shape,
                            kernel_size=patch_size,
                            stride=patch_size),
                                        
            torch.nn.Flatten(start_dim=2,
                                end_dim=3))
        
        self.class_token= torch.nn.Parameter(torch.randn(1,1,
                                                         output_shape),
                                                         requires_grad=True)
        
        self.positional_embedding= torch.nn.Parameter(torch.randn(1,int(number_of_patches+1),
                                                                  output_shape),
                                                                  requires_grad=True)
        
        self.embedding_dropout= torch.nn.Dropout(p=embedding_dropout)
        
    def forward(self,x):
        image_resolution = x.shape[-1]
        batch_size = x.shape[0]
        assert image_resolution % self.patch_size==0, f"Input image size must be divisble by patch size, image size: {image_resolution}, patch size: {self.patch_size}"
        fwd= self.input_layer_stack(x)
        change_dim= fwd.permute(0,2,1)
        class_token = self.class_token.expand(batch_size, -1, -1)
        embedding_with_class_token= torch.cat((class_token,change_dim),dim=1)
        embedding_with_pos_embedding = embedding_with_class_token + self.positional_embedding
        embedding_dropout= self.embedding_dropout(embedding_with_pos_embedding)
       
        return embedding_dropout
    

# Lets create the Multi Head Attention Block 
class MultiHeadAttentionBlock(torch.nn.Module):

    def __init__(self,embedding_dim:int,num_heads:int,attention_dropout:int):
        super().__init__()

        # Layer-Norm (LM) layer
        self.layer_norm= torch.nn.LayerNorm(normalized_shape=embedding_dim)

        # Multi Head Attention Layer
        self.multihead_attn= torch.nn.MultiheadAttention(embed_dim=embedding_dim,
                                                         num_heads=num_heads,
                                                         dropout=attention_dropout,
                                                         batch_first=True)
        
    def forward(self,x):
        # Forward Method
        x= self.layer_norm(x)
        msa_output,_= self.multihead_attn(query=x,
                                          key=x,
                                          value=x,
                                          need_weights=False)

        return msa_output

## Let's create the MLP block

class MLPBlock(torch.nn.Module):
    def __init__(self,embedding_dim:int,mlp_size:int,dropout:int):

        super().__init__()
        # LN layer
        self.layer_norm= torch.nn.LayerNorm(normalized_shape=embedding_dim)
        # MLP layer with GELU and dropout
        self.mlp_layer= torch.nn.Sequential(torch.nn.Linear(in_features=embedding_dim,out_features=mlp_size),
                                      torch.nn.GELU(),
                                      torch.nn.Dropout(p=dropout),
                                      torch.nn.Linear(in_features=mlp_size,out_features=embedding_dim),
                                      torch.nn.Dropout(p=dropout)
                                      )
    # forward method
    def forward(self,x):
        x= self.layer_norm(x)
        x= self.mlp_layer(x)
        return x
    
class TransformerEncoderBlock(torch.nn.Module):

    def __init__(self,embedding_dim:int,num_heads:int,attention_dropout:int,mlp_size:int,mlp_dropout:int):
        super().__init__()

        self.msa = MultiHeadAttentionBlock(embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            attention_dropout=attention_dropout)
        
        self.mlp= MLPBlock(embedding_dim=embedding_dim,
                           mlp_size=mlp_size,
                           dropout=mlp_dropout)
        
    def forward(self,x):
        x= self.msa(x) +x
        x= self.mlp(x) +x
        return x

### Creating the entire ViT architecture

class ViT(torch.nn.Module):

    def __init__(self,input_shape,embedding_dimension,
                 patch_size,number_of_patches,num_heads,
                 attention_dropout,embedding_dropout,mlp_size,
                 dropout,num_output_classes,num_transformer_layers):
        
        super().__init__()
        
        # Input Embedding Layer

        self.input_embedding_layer= input_embedding_layer(input_shape=input_shape,
                                                          output_shape=embedding_dimension,
                                                          patch_size=patch_size,
                                                          number_of_patches=number_of_patches,
                                                          embedding_dropout=embedding_dropout)
        
        self.transformer_encoder_block= torch.nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dimension,
                                                                 num_heads=num_heads,
                                                                 attention_dropout=attention_dropout,
                                                                 mlp_size=mlp_size,mlp_dropout=dropout) for _ in range(num_transformer_layers)])
        
        self.classifier = torch.nn.Sequential(torch.nn.LayerNorm(normalized_shape=embedding_dimension),
                                              torch.nn.Linear(in_features=embedding_dimension,
                                                              out_features=num_output_classes))
        
    def forward(self,x):

        x= self.input_embedding_layer(x)
        x= self.transformer_encoder_block(x)
        x= self.classifier(x[:,0])

        return x
import torch
import torch.nn as nn
from einops import rearrange


class MultiSNN(nn.Module):
    """ Block with multiple SNNs (Ensemble). """
    def __init__(self, in_dims, out_dim):
        super().__init__()
        multi_snn_network = []
        for input_dim in in_dims:
            snn_network = [SNN_Block(input_dim, out_dim), SNN_Block(out_dim, out_dim)]
            multi_snn_network.append(nn.Sequential(*snn_network))
            
        self.net = nn.ModuleList(multi_snn_network)
        
    def forward(self, x):
        outputs = []
        for i, module in enumerate(self.net):
            outputs.append(module(x[i]).float())  
        return torch.stack(outputs, dim=1)
    

class PrototypeAggregator(nn.Module):
    """ Layer that learns to aggregate prototypes. """
    def __init__(self, embedding_dim, num_prototypes):
        super(PrototypeAggregator, self).__init__()
        self.embedding_dim = embedding_dim 
        self.num_prototypes = num_prototypes  

        # Learnable linear layer to produce scalar weights for prototypes
        self.weight_generator = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings, dim):
        """
        :param embeddings: Tensor of shape [B, 73, 144]
        :return: Aggregated embeddings of shape [B, 144]
        """
        # Generate weights for each prototype (shape: [B, 73, 1])
        weights = self.weight_generator(embeddings)  # [B, 73, 1]

        # Normalize weights across the 73 prototypes using softmax
        normalized_weights = nn.functional.softmax(weights, dim=dim)  # [B, 73, 1]

        # Perform weighted sum across the prototype dimension
        weighted_sum = (embeddings * normalized_weights).sum(dim=dim)  # [B, 144]

        return weighted_sum

class SNN_Block(nn.Module):
    """
    Multilayer Reception Block with Self-Normalization (Self Normalizing Network)
    """
    def __init__(self, in_dim, out_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.ELU(), 
            nn.AlphaDropout(p=dropout, inplace=False))

    def forward(self, x):
        return self.net(x)

class CrossAttentionLayer(nn.Module):
    """
    Single attention layer in the attention module
    """

    def __init__(
            self,
            dim=512,
            dim_head=64,
            heads=1,
    ):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)

    
    def forward(self, x, y, return_attention=False):
        x_norm = self.norm_x(x)
        y_norm = self.norm_y(y)

        # derive query, keys, values 
        q = self.to_q(x_norm)
        k = self.to_k(y_norm)
        v = self.to_v(y_norm)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # regular transformer scaling
        q = q * self.scale

        einops_eq = '... i d, ... j d -> ... i j'
        pre_soft_attn_matrix = torch.einsum(einops_eq, q, k)

        attn_matrix = pre_soft_attn_matrix.softmax(dim=-1)

        out  = attn_matrix @ v

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)

        if return_attention:
            return out, attn_matrix.squeeze().detach().cpu()
    
        return out

       
class FeedForward(nn.Module):
    """ Feedforward Neural Network block. """
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim), int(dim))
        )

    def forward(self, x):
        return self.net(self.norm(x))

class FeedForwardEnsemble(nn.Module):
    """Block with multiple FNNs (Ensemble). """
    def __init__(self, dim, dropout=0., num=16):
        super().__init__()
        self.num = num
        self.net = nn.ModuleList([FeedForward(dim, dropout) for _ in range(num)])

    def forward(self, x):
        """
        Args:
            x: (B, proto, d)
        """
        assert x.shape[1] == self.num
        out = []
        for idx in range(self.num):
            out.append(self.net[idx](x[:,idx:idx+1,:]))
        out = torch.cat(out, dim=1)

        return out

    




                









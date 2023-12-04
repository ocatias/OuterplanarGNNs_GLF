
import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims = None):
        super(NodeEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()
        if feature_dims is None:
            feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        x = x.long()
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

# self.cat_feats = []
#         self.num_feats = []
#         self.mlps_for_num_feats = []

#         for i, dim in enumerate(feature_dims):
#             if dim > 0:
#                 emb = torch.nn.Embedding(dim, emb_dim)
#                 torch.nn.init.xavier_uniform_(emb.weight.data)
#                 self.atom_embedding_list.append(emb)
#                 self.cat_feats.append(i)
#             else:
#                 self.mlps_for_num_feats.append(torch.nn.Linear(1, emb_dim, bias=False))
#                 self.num_feats.append(i)
                
#     def forward(self, x):
#         x_embedding = 0
#         x = x.long()
        
#         for i, dim in enumerate(self.cat_feats):
#             x_embedding += self.atom_embedding_list[i](x[:,dim])
            
#         for i, dim in enumerate(self.num_feats):
#             print("NUMERICAL FEATUREWS")
#             x_embedding += self.mlps_for_num_feats[i](x[:,dim])

#         return x_embedding
class EdgeEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim, activation, feature_dims = None):
        super(EdgeEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        if feature_dims is None:
            feature_dims = get_bond_feature_dims()
            
        print(f"edge feature_dims: {feature_dims}")
        
        self.cat_feats = []
        self.num_feats = []
        
        self.mlps_for_num_feats = torch.nn.ModuleList([])

        for i, dim in enumerate(feature_dims):
            if dim > 0:
                emb = torch.nn.Embedding(dim, emb_dim)
                torch.nn.init.xavier_uniform_(emb.weight.data)
                self.bond_embedding_list.append(emb)
                self.cat_feats.append(i)
            else:
                self.mlps_for_num_feats.append(torch.nn.Sequential(
                    torch.nn.Linear(1, 4),
                    activation,
                    torch.nn.Linear(4, 16),
                    activation,
                    torch.nn.Linear(16, emb_dim),
                    activation
                ))
                self.num_feats.append(i)
        
    def forward(self, edge_attr):
        bond_embedding = 0   
        for i, dim in enumerate(self.cat_feats):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,dim])
            
        for i, dim in enumerate(self.num_feats):
            bond_embedding += self.mlps_for_num_feats[i](torch.unsqueeze(edge_attr[:,dim], dim = 1).float())
            

        return bond_embedding   


class EgoEncoder(torch.nn.Module):
    # From ESAN
    def __init__(self, encoder):
        super(EgoEncoder, self).__init__()
        self.num_added = 2
        self.enc = encoder

    def forward(self, x):
        return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:])))


class ZincAtomEncoder(torch.nn.Module):
    # From ESAN
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus':
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, ReLU, Dropout, Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
import torch_scatter
from torch_sparse import SparseTensor, set_diag

class baseGATv2(MessagePassing):
    def __init__(self, in_channels, out_channels, heads = 1,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(baseGATv2, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None
        self._alpha = None
        # self.lin_l is the linear transformation that you apply to embeddings
        # BEFORE message passing.
        self.lin_l =  Linear(in_channels, heads*out_channels)
        self.lin_r = self.lin_l

        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        self.reset_parameters()

    #initialize parameters with xavier uniform
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, size = None):

        H, C = self.heads, self.out_channels # DIM：H, outC
        #Linearly transform node feature matrix.
        x_source = self.lin_l(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]
        x_target = self.lin_r(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]

        #  Start propagating messages (runs message and aggregate)
        out= self.propagate(edge_index, x=(x_source,x_target),size=size) # DIM: [nodes, H, outC]
        out= out.view(-1, self.heads * self.out_channels)       # DIM: [nodes, H * outC]
        alpha = self._alpha
        self._alpha = None
        return out

    #Process a message passing
    def message(self, x_j,x_i,  index, ptr, size_i):
        #computation using previous equationss
        x = x_i + x_j
        x  = F.leaky_relu(x, self.negative_slope)   # See Equation above: Apply the non-linearty function
        alpha = (x * self.att).sum(dim=-1)          # Apply attnention "a" layer after the non-linearity
        alpha = softmax(alpha, index, ptr, size_i)  # This softmax only calculates it over all neighbourhood nodes
        self._alpha = alpha
        alpha= F.dropout(alpha,p=self.dropout,training=self.training)
        # Multiple attention with node features for all edges
        out= x_j*alpha.unsqueeze(-1)

        return out
    #Aggregation of messages
    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim,
                                    dim_size=dim_size, reduce='sum')
        return out
class GATv2modif(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,args):
        super(GATv2modif, self).__init__()
        #use our gat message passing
        self.args = args
#Added
        self.conv1 = baseGATv2(input_dim, hidden_dim,heads=args['heads'])
        self.conv2 = baseGATv2(args['heads'] *hidden_dim, hidden_dim,heads=args['heads'])

        self.post_mp = nn.Sequential(
            nn.Linear(args['heads']  * hidden_dim, hidden_dim), nn.Dropout(args['dropout'] ),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        #x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        #added
        x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        #x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)
        # Added

        # MLP output
        x = self.post_mp(x)
        return F.sigmoid(x)

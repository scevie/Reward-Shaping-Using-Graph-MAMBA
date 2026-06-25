import torch
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
from sat.data import GraphDataset
from sat import GraphTransformer

# Load the ZINC dataset using our wrapper GraphDataset,
# which automatically creates the fully connected graph.
# For datasets with large graph, we recommend setting return_complete_index=False
# leading to faster computation
dset = datasets.ZINC('./datasets/ZINC', subset=True, split='train')
dset = GraphDataset(dset)

# Create a PyG data loader
train_loader = DataLoader(dset, batch_size=16, shuffle=True)

# Create a SAT model
dim_hidden = 16
gnn_type = 'gcn' # use GCN as the structure extractor
k_hop = 2 # use a 2-layer GCN

model = GraphTransformer(
    in_size=28, # number of node labels for ZINC
    num_class=1, # regression task
    d_model=dim_hidden,
    dim_feedforward=2 * dim_hidden,
    num_layers=2,
    batch_norm=True,
    gnn_type='gcn', # use GCN as the structure extractor
    use_edge_attr=True,
    num_edge_features=4, # number of edge labels
    edge_dim=dim_hidden,
    k_hop=k_hop,
    se='gnn', # we use the k-subtree structure extractor
    global_pool='cls'
)

for data in train_loader:
    output = model(data) # batch_size x 1
    print(output.shape)
    break
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

#a simple Graph Neural Network
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

#generate a graph
def create_sample_graph():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
    ], dtype=torch.long)  # Edge list
    
    x = torch.rand((6, 3))  # Random node features (6 nodes, 3 features each)
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)  # Node labels
    return Data(x=x, edge_index=edge_index, y=y)

#plot function for graph
def plot_graph(data, title, labels=None):
    G = nx.Graph()
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color=labels, cmap=plt.cm.coolwarm, node_size=500, edge_color='gray')
    plt.title(title)
    plt.show()
    plt.savefig('GNN_Final.png')
    print("The Graph is saved")

#training the GNN
def train():
    data = create_sample_graph()
    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    plot_graph(data, "Input Graph", labels=data.y.numpy())
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    pred = model(data).argmax(dim=1).detach().numpy()
    plot_graph(data, "Predicted Graph", labels=pred)
    
    print("Final Predictions:", pred)

if __name__ == "__main__":
    train()
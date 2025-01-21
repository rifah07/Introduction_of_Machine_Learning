import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Define the GNN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Create a simple graph
def create_sample_graph():
    # Define the edges of the graph (source, target)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 3],  # Source nodes
        [1, 0, 2, 1, 4, 5]   # Target nodes
    ], dtype=torch.long)

    # Define node features (6 nodes, 3 features per node)
    x = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=torch.float)

    # Define node labels (for classification)
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# Plot the graph using networkx
def plot_graph(data, title, labels=None, node_color=None):
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G)

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_color, cmap=plt.cm.Paired, node_size=500, edge_color="gray")

    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in range(len(labels))}, font_color='white')

    plt.title(title)
    plt.show()

# Train the GNN model
def train():
    # Create the graph data
    data = create_sample_graph()

    # Define the model
    input_dim = data.x.size(1)
    hidden_dim = 16
    output_dim = 2  # Number of classes
    model = GCN(input_dim, hidden_dim, output_dim)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Plot the input graph
    plot_graph(data, title="Input Graph", labels=data.y.tolist(), node_color=data.y.tolist())

    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    _, pred = model(data).max(dim=1)

    # Plot the predicted graph
    plot_graph(data, title="Predicted Graph", labels=pred.tolist(), node_color=pred.tolist())

    # Plot actual vs. predicted comparison
    plt.figure(figsize=(8, 6))
    plt.plot(data.y.numpy(), label="Actual", marker='o')
    plt.plot(pred.numpy(), label="Predicted", marker='x')
    plt.xlabel("Node Index")
    plt.ylabel("Class")
    plt.title("Actual vs Predicted Node Classes")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the training
if __name__ == "__main__":
    train()
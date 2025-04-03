import torch
import torch.nn as nn


class GaussianEmbedding(nn.Module):
    def __init__(self, num_terms, num_channels):
        super(GaussianEmbedding, self).__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels

        self.h = nn.Parameter(torch.randn(num_terms + 1, num_channels))
        nn.init.xavier_uniform_(self.h)

    def forward(self, A):
        batch_size, num_nodes, _ = A.shape
        Y_hat = torch.zeros(batch_size, num_nodes, self.num_channels, device=A.device)
        for c in range(self.num_channels):
            result = self.h[0, c] * torch.eye(num_nodes, device=A.device).unsqueeze(0).expand(batch_size, -1, -1)
            for i in range(1, self.num_terms + 1):
                A_power_i = torch.matrix_power(A, i)
                result += self.h[i, c] * A_power_i
            Y_hat[..., c] = torch.diagonal(result, dim1=-2, dim2=-1)
        return Y_hat

class EigenEmbedding(nn.Module):
    def __init__(self):
        super(EigenEmbedding, self).__init__()

    def forward(self, A):
        eigenvectors = []
        for i in range(A.shape[0]):
            _, V = torch.linalg.eigh(A[i])
            eigenvectors.append(V)
        return torch.stack(eigenvectors, dim=0)


class GraphConvolutionLayer(nn.Module):
    def __init__(self, num_terms, num_channels):
        super(GraphConvolutionLayer, self).__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels

        self.H = nn.Parameter(torch.randn(num_terms + 1, num_channels, num_channels))
        nn.init.xavier_uniform_(self.H)

        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, A, X):
        Y_hat = X @ self.H[0] 
        for i in range(1, self.num_terms + 1):
            A_power_i = torch.matrix_power(A, i)
            Y_hat += torch.bmm(A_power_i, X) @ self.H[i]
        Y_hat = self.layer_norm(Y_hat)
        Y_hat = self.activation(Y_hat)
        return Y_hat


class NodeVarGraphConvolutionLayer(nn.Module):
    def __init__(self, num_terms, num_channels, num_nodes):
        super(NodeVarGraphConvolutionLayer, self).__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels
        self.num_nodes = num_nodes

        self.h = nn.Parameter(torch.randn(num_terms + 1, num_channels, num_nodes))
        nn.init.xavier_uniform_(self.h)

        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, A, X):
        batch_size, num_nodes, num_channels_in = X.shape
        Y_hat = torch.zeros(batch_size, num_nodes, self.num_channels, device=A.device)
        for c in range(self.num_channels):
            result = torch.zeros(batch_size, num_nodes, device=A.device)
            h_diag = torch.diag_embed(self.h[0, c])
            A_w = h_diag @ torch.eye(num_nodes, device=A.device).unsqueeze(0).expand(batch_size, -1, -1)
            for ch in range(num_channels_in):
                result += torch.bmm(A_w, X[..., ch].unsqueeze(-1)).squeeze(-1)

            for i in range(1, self.num_terms + 1):
                A_power_i = torch.matrix_power(A, i)
                h_diag = torch.diag_embed(self.h[i, c])
                A_w = h_diag @ A_power_i
                for ch in range(num_channels_in):
                    result += torch.bmm(A_w, X[..., ch].unsqueeze(-1)).squeeze(-1)
            Y_hat[..., c] = result
        Y_hat = self.layer_norm(Y_hat)
        Y_hat = self.activation(Y_hat)
        return Y_hat


class GNN(nn.Module):
    def __init__(self, num_layers, num_terms=3, feature_dim=10):
        super(GNN, self).__init__()

        # self.embedding_layer = GaussianEmbedding(num_terms, feature_dim)
        self.embedding_layer = EigenEmbedding()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConvolutionLayer(num_terms, feature_dim))

        self.out_x = nn.Linear(feature_dim, feature_dim)
        self.out_y = nn.Linear(feature_dim, feature_dim)

    def forward(self, A):
        Z = self.embedding_layer(A)
        for layer in self.layers:
            Z = layer(A, Z)
        X = self.out_x(Z)
        Y = self.out_y(Z)
        outer = torch.bmm(X, Y.transpose(1, 2))
        return torch.sigmoid(outer)


class NodeVarGNN(nn.Module):
    def __init__(self, num_layers, num_terms=3, feature_dim=10):
        super(NodeVarGNN, self).__init__()

        # self.embedding_layer = GaussianEmbedding(num_terms, feature_dim)
        self.embedding_layer = EigenEmbedding()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(NodeVarGraphConvolutionLayer(num_terms, feature_dim, feature_dim))

        self.out_x = nn.Linear(feature_dim, feature_dim)
        self.out_y = nn.Linear(feature_dim, feature_dim)

    def forward(self, A):
        Z = self.embedding_layer(A)
        for layer in self.layers:
            Z = layer(A, Z)
        X = self.out_x(Z)
        Y = self.out_y(Z)
        outer = torch.bmm(X, Y.transpose(1, 2))
        return torch.sigmoid(outer)

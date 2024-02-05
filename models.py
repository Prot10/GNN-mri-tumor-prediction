import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, BatchNorm, GraphConv, GATConv
import geoopt
import torch.nn as nn



####################################    
### Convolutional Neural Network ###
####################################

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



####################################    
### Graph Convolution Network v1 ###
####################################

class GCNv1(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, num_classes):
        super(GCNv1, self).__init__()
        torch.manual_seed(123)
        self.conv = GraphConv(feature_size, hidden_channels)
        self.fc = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv(x, edge_index))
        
        x = global_mean_pool(x, batch)
        
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    

####################################    
### Graph Convolution Network v2 ###
####################################

class GCNv2(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, num_classes):
        super(GCNv2, self).__init__()
        self.embed = Linear(feature_size, hidden_channels)
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.fc1 = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.embed(x))
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = x + identity
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = x + identity
        
        x = global_mean_pool(x, batch)
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
####################################    
### Graph Convolution Network v3 ###
####################################

class GCNv3(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, num_classes):
        super(GCNv3, self).__init__()
        self.fc1 = Linear(feature_size, hidden_channels)
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.fc3 = Linear(hidden_channels, hidden_channels)
        self.fc4 = Linear(hidden_channels, hidden_channels // 2)
        self.fc5 = Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.conv2(x, edge_index))
        
        x = global_mean_pool(x, batch)
        
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
##################################    
### Graph Attention Network v1 ###
##################################

class GATv1(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, num_classes):
        super(GATv1, self).__init__()
        self.embed = torch.nn.Linear(feature_size, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = torch.nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.fc3 = torch.nn.Linear(hidden_channels// 4, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.embed(x))
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = x + identity
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = x + identity
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = x + identity
        
        x = global_mean_pool(x, batch)
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



##################################    
### Graph Attention Network v2 ###
##################################

class GATv2(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, num_classes, c=1.0):
        super(GATv2, self).__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.embed = torch.nn.Linear(feature_size, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = torch.nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.fc3 = torch.nn.Linear(hidden_channels // 4, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.embed(x))
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = x + identity
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = x + identity
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = x + identity

        # Project points to Poincaré ball
        x = self.manifold.expmap0(x)
        x = global_mean_pool(x, batch)
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
##################################    
### Graph Attention Network v3 ###
##################################
    
class GATv3(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, num_classes, c=1.0, dropout_rate=0.5):
        super(GATv3, self).__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.embed = torch.nn.Linear(feature_size, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc3 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc4 = torch.nn.Linear(hidden_channels // 2, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.embed(x))
        identity = x

        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = x + identity
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))

        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = x + identity
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))

        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = x + identity
        x = self.dropout(x)

        x = self.manifold.expmap0(x)
        x = global_mean_pool(x, batch)
        
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
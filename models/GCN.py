import torch
import torch_geometric.nn as nnGeo
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import GATConv, LayerNorm, GATv2Conv
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool, TopKPooling
import numpy as np



class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, pooling = 'mean', node_classification = False):
        super().__init__()
        self.node_classification = node_classification
        
        if pooling is None:
            self.pooling = None
        else:
            self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool}[pooling]
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels,hidden_channels)

        if pooling is not None:
            # use GCN to classify
            self.lin = nn.Linear(hidden_channels, num_classes)
        
        if node_classification and pooling is not None:
            self.top_k_pool = TopKPooling(hidden_channels)
       
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        if self.pooling is None:
            return x
       
        if self.pooling is not None and not self.node_classification:
            x3 = self.pooling(x, data.batch)   
            x3 = F.dropout(x3, p=0.25, training=self.training)
            x3 = self.lin(x3)
            return x3
        
        elif self.pooling is not None and self.node_classification:
            x3 = F.dropout(x, p=0.25, training=self.training)
            
            node_predictions = self.lin(x3) #num_nodes x num_classes
          
            pooled_x, edge_index, edge_attr, batch, _, _ = self.top_k_pool(node_predictions, data.edge_index, None, data.batch)
            
            x_final = self.pooling(pooled_x, data.batch)
            
            return x_final, node_predictions

        

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, num_classes, pooling = 'max', heads = 1, node_classification = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.node_classification = node_classification

        self.gat1 = GATConv(input_dim, hidden_channels, heads=heads)
        self.bn1 =torch.nn.GroupNorm(16,hidden_channels * heads)#, track_running_stats=True)
        self.gat2 = GATConv(hidden_channels*heads, hidden_channels*heads, heads=heads)
        self.bn2 =torch.nn.GroupNorm(16,hidden_channels * heads)#, track_running_stats=True)
        self.gat3 = GATConv(hidden_channels*heads, hidden_channels*heads, heads=heads)
        self.bn3 =torch.nn.GroupNorm(16,hidden_channels * heads)#, track_running_stats=True)
        self.lin = nn.Linear(hidden_channels*heads , num_classes)  
        self.node_classification = node_classification
        
        if pooling is None:
            self.pooling = None
        else:
            self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool}[pooling]
        #self.ln = LayerNorm(hidden_channels)
        if node_classification and pooling is not None:
            self.top_k_pool = TopKPooling(num_classes)
       
        
    def forward(self, data):
    
        x, edge_index= data.x, data.edge_index        
        x = self.gat1(x, edge_index) 
        #x = self.bn1(x)
        x = x.relu()
       
        x = self.gat2(x, edge_index)
        #x = self.bn2(x)
        x = x.relu()
        
        
        x = self.gat3(x, edge_index)
        #x = self.bn3(x) 


        if self.pooling is None:
            return x
       
        
        if self.pooling is not None and not self.node_classification:
            x3 = self.pooling(x, data.batch)   
            x3 = F.dropout(x3, p=0.25, training=self.training)
            x3 = self.lin(x3)
            return x3
        
        elif self.pooling is not None and self.node_classification:
            x3 = F.dropout(x, p=0.25, training=self.training)
            node_predictions = self.lin(x3) #num_nodes x num_classes
            pooled_x, edge_index, edge_attr, batch, _, _ = self.top_k_pool(node_predictions, data.edge_index, None, data.batch)
            x_final = self.pooling(pooled_x, batch)
            
            return x_final, node_predictions
        
        
        
class LinearGraphClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, pooling='mean', node_classification = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.node_classification = node_classification
    
        
        # Simple linear classifier
        self.lin = nn.Linear(input_dim, num_classes)

        if pooling is None:
            self.pooling = None
        else:
            self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool}[pooling]
        if node_classification and pooling is not None:
            self.top_k_pool = TopKPooling(num_classes)
        
    def forward(self, data):
        x, batch = data.x, data.batch
        
        # For node classification
        node_predictions = self.lin(x)
        pooled_x, edge_index, edge_attr, batch, _, _ = self.top_k_pool(node_predictions, data.edge_index, None, data.batch)
        x_final = self.pooling(pooled_x, batch)

        
        return x_final, node_predictions
        

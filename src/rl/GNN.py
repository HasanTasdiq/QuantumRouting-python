import tensorflow as tf
from keras import Model, layers, Sequential
class QRoutingGATFlat(Model):
    def __init__(self, num_nodes, input_flat_dim, hidden_dim=64, num_heads=4):
        super(QRoutingGATFlat, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 1. Input Adapter (The "Learned Un-Flatten")
        # Takes the huge flat input (Graph + Dist + Embeddings) and projects it
        # into a shape that represents (Num_Nodes, Hidden_Dim).
        # We use a bottleneck (1024) to keep parameter count sane.
        self.input_flatten = layers.Flatten()
        self.projection_1 = layers.Dense(1024, activation='relu') 
        self.projection_2 = layers.Dense(num_nodes * hidden_dim, activation='relu') 
        
        # 2. Graph Attention Mechanism (The GNN Core)
        # Now that we have data for each node, they "talk" to each other.
        self.attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
        self.layernorm1 = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)
        
        # 3. Feed Forward Network (Standard Transformer Block)
        # Helps process the attended information
        self.ffn = Sequential([
            layers.Dense(hidden_dim, activation='relu'), 
            layers.Dense(hidden_dim)
        ])
        self.layernorm2 = layers.LayerNormalization()

        # 4. Output Head
        # Projects each node's features down to 1 Q-value per node
        self.q_output = layers.Dense(1, activation='linear')

    def call(self, inputs):
        # inputs shape: [Batch, Input_Dim]
        
        # Step A: Project Flat Vector -> Graph Structure
        x = self.input_flatten(inputs)
        x = self.projection_1(x)
        x = self.projection_2(x)
        
        # Reshape to [Batch, Num_Nodes, Hidden_Dim]
        # This creates "Node Embeddings" from your flat graph data
        node_embeddings = tf.reshape(x, (-1, self.num_nodes, self.hidden_dim))
        
        # Step B: Apply Graph Attention
        # Nodes attend to each other to decide routing
        attended = self.attention_layer(query=node_embeddings, value=node_embeddings, key=node_embeddings)
        
        # Add & Norm (Skip Connection)
        x_att = self.layernorm1(node_embeddings + attended)
        
        # Step C: Feed Forward & Residual
        ffn_out = self.ffn(x_att)
        x_final = self.layernorm2(x_att + ffn_out)
        
        # Step D: Generate Q-Values
        # Shape: [Batch, Num_Nodes, 1]
        q_vals = self.q_output(x_final)
        
        # Step E: Flatten to match your expected output: [Batch, Num_Nodes]
        return tf.squeeze(q_vals, axis=-1)
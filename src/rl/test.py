import tensorflow as tf
from keras.layers import Embedding, Flatten, Attention
import numpy

# Create a sample 2D input
input_array = tf.constant([[0,0,0,1,0,1,0,0,0,0],[0,1,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,1,0]])

# Create an Embedding layer
embedding_layer = Embedding(input_dim=100, output_dim=1)

# Apply the Embedding layer
embedded_output = embedding_layer(input_array)

# Flatten the output
flattened_output = Flatten()(embedded_output)


attention = Attention()


# Print the shape of the flattened output
# print(embedded_output)
# print(flattened_output)
out1 = Flatten()(tf.constant([numpy.array(flattened_output)]))[0]
out1 = embedded_output
print(numpy.array(embedded_output))
print('-------------------')
# outa = attention([out1, out1, out1])


input_vector = tf.constant(out1, dtype=tf.float32)

# Expand dimensions to simulate a sequence
# Shape: (batch_size=1, seq_length=1, embedding_dim=6)
input_vector = tf.expand_dims(tf.expand_dims(input_vector, axis=0), axis=0)

# Create query, key, and value as the same (self-attention)
query = key = value = input_vector  # Shape: (batch_size=1, seq_length=1, embedding_dim=6)

# Initialize the Attention layer
attention_layer = Attention()

# Compute attention
attention_output = attention_layer([query, key, value])

print('out1 : ' , out1.numpy())
print("Input Vector:", query.numpy())
print("Attention Output:", attention_output.numpy())
# print(outa)


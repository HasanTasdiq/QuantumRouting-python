import tensorflow as tf
from keras.layers import Embedding, Flatten, Attention
import numpy

# Create a sample 2D input
input_array = tf.constant([[0, 1, 1,0],
[0, 1, 0, 1],
[1, 0, 0, 1]])

# Create an Embedding layer
embedding_layer = Embedding(input_dim=10, output_dim=1)

# Apply the Embedding layer
embedded_output = embedding_layer(input_array)

# Flatten the output
flattened_output = Flatten()(embedded_output)


attention = Attention(use_scale=True)


# Print the shape of the flattened output
# print(embedded_output)
# print(flattened_output)
out1 = Flatten()(tf.constant([numpy.array(flattened_output)]))[0]
out1 = embedded_output
# print(numpy.array(embedded_output))
# print('-------------------')
# outa = attention([out1, out1, out1])


input_vector = tf.constant(out1, dtype=tf.float32)

# Expand dimensions to simulate a sequence
# Shape: (batch_size=1, seq_length=1, embedding_dim=6)
# input_vector = tf.expand_dims(tf.expand_dims(input_vector, axis=0), axis=0)

# Create query, key, and value as the same (self-attention)
query = key = value = input_vector  # Shape: (batch_size=1, seq_length=1, embedding_dim=6)

# Initialize the Attention layer
attention_layer = Attention()

# Compute attention
attention_output = attention_layer([query, key, value])

# print('embedded out1 : ' , out1.numpy())
# print("Input Vector:", query.numpy())
# print("Attention Output:", attention_output.numpy())




def get_embedded_output(inp , formatted = False):

        # Apply the Embedding layer
        embedded_output = embedding_layer(tf.constant(inp))
        # print('------embedded_output-----' , tf.constant(embedded_output, dtype=tf.float32))
        print('------embedded_output-----' , embedded_output)
        flattened_output = Flatten()(embedded_output)
        print('------flattened_output-----' , flattened_output)

        # Print the shape of the flattened output
        # print(embedded_output)
        # print(flattened_output)

        out1 = Flatten()(tf.constant([numpy.array(flattened_output)]))[0]
        print('------out1-----' , out1)

        out1 = embedded_output
        if formatted:
            return list(out1.numpy())
        return out1
    
def get_emb_attention(  inp):
        atten_input = get_embedded_output(inp)
        print('----embedded--- ')
        print(atten_input)
        input_vector = tf.constant(atten_input, dtype=tf.float32)

        # Expand dimensions to simulate a sequence
        # Shape: (batch_size=1, seq_length=1, embedding_dim=6)
        # input_vector = tf.expand_dims(tf.expand_dims(input_vector, axis=0), axis=0)

        # Create query, key, and value as the same (self-attention)
        query = key = value = input_vector  # Shape: (batch_size=1, seq_length=1, embedding_dim=6)

        # Initialize the Attention layer

        # Compute attention
        attention_output = attention_layer([query, key, value])

        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        # print(attention_output)
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        return attention_output.numpy()

print('----from funct')
print(get_embedded_output(input_array))

a = b = c = 10
# print(a,b,c)

# print('----from funct   222222')
# print(get_emb_attention(input_array))
# print(outa)


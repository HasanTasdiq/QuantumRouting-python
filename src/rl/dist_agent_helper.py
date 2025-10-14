from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Flatten, Attention, Dense, MultiHeadAttention, LayerNormalization



executor = ProcessPoolExecutor(max_workers=50)
SIZE = 100
embedding_layer = Embedding(input_dim=20, output_dim=1)
attention_layer = Attention()
dense_proj = Dense(64, activation='relu')
mha = MultiHeadAttention(num_heads=4, key_dim=16)
ln = LayerNormalization()
_concat_buffer = None

def process_update_action(  p):
        # print('process_update_action called ' , p.reqIndex)
    # Recreate necessary agent if needed or call static function
        return update_action(
            p.reqIndex,
            p.current_node_id,
            p.next_node_id,
            p.current_state,
            p.done_episode,
            p.timeSlot,
            p.reward,
            p.node_matrix,
            p.req_matrix,
            p.dist_matrix
        )

def update_action( request_index ,current_node_id,  action  , current_state  , done , timeSlot,lreward,ent_matrix , req_matrix,dist_matrix):
        # print('update_action called ')
        request = req_matrix[request_index][:6]
        request[3] = req_matrix[request_index + len(req_matrix)//2]
        prev_ent_matrix = current_state[0]
        prev_req_matrix = current_state[1]
        prev_dist_matrix = dist_matrix
        prev_request = prev_req_matrix[request_index][:6]
        prev_request[3] = prev_req_matrix[request_index + len(prev_req_matrix)//2]

        # print('update_action got request ')
        current_state = schedule_routing_state_dist(prev_request , prev_ent_matrix , prev_req_matrix, prev_dist_matrix)

        # print('doooooooooooooooooooone -------------- ' , done , (request[0].id , request[1].id) ,current_node_id , action)
        if not done:
            t = time.time()
            next_state = schedule_routing_state_dist(request, ent_matrix , req_matrix,dist_matrix)
            # print('update action get state time ' , time.time()-t)
        else:
            next_state = None

        if next_state is None:
            # print('next state is none in update action!!!!!!!!!!!!!')
            next_state = current_state
        # done = False
        t = time.time()
        mask = get_mask_one_req_schedule_route(request,ent_matrix , req_matrix) #action is the next node id
        # print('update action get get_mask_shcedule_route time ' , time.time()-t)
        t = time.time()
        data = (request , action , timeSlot ,current_node_id,  current_state , next_state ,mask ,  done, lreward)
        return data


def get_mask_one_req_schedule_route(reqState , ent_matrix=None, req_matrix=None):
        mask = [None for _ in range(SIZE)]
        

        src,dst,current_node_id , path , index , done = reqState
        state_graph = ent_matrix

        neighbors = [i for i, x in enumerate(state_graph[current_node_id]) if x >= 1]
        # print('in get mask +++==== ' , src.id,dst.id,current_node.id , neighbors)
        for n in neighbors:
            if not path[n] and n != current_node_id:
                mask[ n] = 1

        if not mask.count(1): #make a random mask
            mask = [1 for _ in range(SIZE)]

        return np.array(mask)
    # === 2. Request embeddings ===
def get_request_embeddings( req_matrix):
        # print('get_request_embeddings called' , len(req_matrix)//2)
        features = []
        for req in req_matrix:
            vec = [0] * SIZE
            if not req[5]:  # if not completed
                vec[req[0]] = 1  # current node
                vec[req[1]] = 10  # destination
            features.append(vec)
        # print('get_request_embeddings before return')
        return tf.convert_to_tensor(features, dtype=tf.float32)


    # === 3. Request-level attention ===
def apply_request_attention( request_tensor):
        # print('called apply_request_attention ')

        # Project to 64D
        proj = dense_proj(request_tensor)  # shape: [num_requests, 64]
        # print('after dense proj ')
        # Add batch dimension
        proj = tf.expand_dims(proj, axis=0)  # shape: [1, num_requests, 64]
        # print('after expand dims ')
        # Apply MHA
        attn_out = mha(query=proj, key=proj, value=proj)  # Correct usage
        # print('after mha ')
        # Residual + LayerNorm
        output = ln(proj + attn_out)  # shape: [1, num_requests, 64]
        # print('after layer norm ')
        return tf.squeeze(output, axis=0)  # shape: [num_requests, 64]

    # === 4. Neighbor embedding ===
def get_neighbor_embeddings( state_graph, current_node_id):
        neighbors = state_graph[current_node_id]
        neighbor_feats = []
        for node_id, has_link in enumerate(neighbors):
            if has_link > 0:
                feat = [0] * SIZE
                feat[node_id] = 1  # one-hot neighbor
                vec = tf.convert_to_tensor(feat, dtype=tf.float32)
                vec = tf.expand_dims(vec, axis=0)  # (1, SIZE)
                vec = Dense(64, activation='relu')(vec)
                vec = tf.squeeze(vec, axis=0)      # (64,)
                neighbor_feats.append(vec)
        return neighbor_feats



    # === 5. Neighbor attention ===
def apply_neighbor_attention( curr_emb, neighbor_embs):
        if not neighbor_embs:
            return np.zeros(64)
        stack = tf.stack(neighbor_embs)  # [num_neighbors, 64]
        query = tf.expand_dims(curr_emb, axis=0)  # [1, 64]
        scores = tf.matmul(query, stack, transpose_b=True) / tf.math.sqrt(64.0)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, stack)[0].numpy()
        return context


    # === 6. Main function ===
def schedule_routing_state_dist( curr_req, ent_matrix=None, req_matrix=None, dist_matrix=None):


        # 1. Link matrices
        # print('in schedule_routing_state_dist' )
        state_graph, state_dist = ent_matrix, dist_matrix
        # print('state_graph found')

        state_graph_flat = np.array(state_graph).flatten()  # shape: [SIZE × SIZE]
        state_dist_flat = np.array(state_dist).flatten()    # shape: [SIZE × SIZE]
        # print('state_graph_flat found')
        # 2. Request embeddings
        # print('going to get get_request_embeddings ')
        req_tensor = get_request_embeddings(req_matrix)
        # print('req_tensor found')

        # 3. Apply self-attention
        # print('going to get apply_request_attention ')
        attn_encoded = apply_request_attention(req_tensor).numpy()
        # print('attn_encoded found')
        # 4. Locate current request
        curr_index = curr_req[4]
        # print('curr_index found' , curr_index)

        curr_emb = attn_encoded[curr_index]

        # 5. Neighbor context
        # print('going to get get_neighbor_embeddings ')
        neighbor_embs = get_neighbor_embeddings(state_graph, curr_req[2])
        # print('neighbor_embs found' , len(neighbor_embs))
        # print('going to get apply_neighbor_attention ')
        context_vec = apply_neighbor_attention(curr_emb, neighbor_embs)
        # print('context_vec found')

        # 6. Local info
        local = [0] * SIZE
        local[curr_req[2]] = 10
        local[curr_req[1]] = 10

        # 7. Final state vector
        ret = np.concatenate([
            curr_emb,         # attention-aware request embedding
            context_vec,      # neighbor context
            np.array(local),   # current and destination
            state_graph_flat,
            state_dist_flat
        ])

        # total_len = curr_emb.size + context_vec.size + len(local) + len(state_graph_flat) + len(state_dist_flat)
        # # ret = np.empty(total_len, dtype=np.float32)
        # global _concat_buffer
        # if _concat_buffer is None or _concat_buffer.size < total_len:
        #     _concat_buffer = np.empty(total_len, dtype=np.float32)
        # ret = _concat_buffer[:total_len]

        # start = 0
        # ret[start:start+curr_emb.size] = curr_emb
        # start += curr_emb.size
        # ret[start:start+context_vec.size] = context_vec
        # start += context_vec.size
        # ret[start:start+len(local)] = local
        # start += len(local)
        # ret[start:start+len(state_graph_flat)] = state_graph_flat
        # start += len(state_graph_flat)
        # ret[start:start+len(state_dist_flat)] = state_dist_flat

        return ret

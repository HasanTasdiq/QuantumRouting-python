from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import pickle
import time
import os

import numpy as np
import redis
import tensorflow as tf
from keras.layers import Embedding, Flatten, Attention, Dense, MultiHeadAttention, LayerNormalization


max_workers = 60
executor = ProcessPoolExecutor(max_workers=max_workers)
SIZE = 100
# embedding_layer = Embedding(input_dim=20, output_dim=1)
# attention_layer = Attention()
dense_proj = Dense(64, activation='relu')
mha = MultiHeadAttention(num_heads=4, key_dim=16)
ln = LayerNormalization()

mpredis = redis.Redis(host='localhost', port=6379, db=0)


# run 25k
# START_EPSILON_DECAYING = 10000
# END_EPSILON_DECAYING = 20000
# REPLAY_MEMORY_SIZE = 80000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 20000  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 1500  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)

# for 5k local
START_EPSILON_DECAYING = 4000
END_EPSILON_DECAYING = 6000
REPLAY_MEMORY_SIZE = 15000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 512  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 70  # Terminal states (end of episodes)

# for 3k local
# START_EPSILON_DECAYING = 1000
# END_EPSILON_DECAYING = 2500
# REPLAY_MEMORY_SIZE = 10000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 3000  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 500  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 70  # Terminal states (end of episodes)

#for 10k local
# START_EPSILON_DECAYING = 3000
# END_EPSILON_DECAYING = 8000
# REPLAY_MEMORY_SIZE = 15000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 512  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)


# for testing
# START_EPSILON_DECAYING = 10
# END_EPSILON_DECAYING = 20
# REPLAY_MEMORY_SIZE = 10000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
# MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)



replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
REPLAY_MEMORY_FILE = "replay_memory.pkl"

def load_model_from_redis( model, model_name="dqrl_model"):
    """Load model weights from Redis if newer version available"""
    try:
        redis_version = int(mpredis.get(f"{model_name}_version") or 0)
        serialized = mpredis.get(f"{model_name}_weights")
            
        if serialized:
            weights = pickle.loads(serialized)
            model.set_weights(weights)
            print(f"Model loaded from Redis - version {redis_version}")
            return redis_version
        return None
    except Exception as e:
        print(f"Error loading from Redis: {e}")
        return None
        
def save_model_to_redis( model, model_name="dqrl_model"):
    """Save model weights to Redis"""
    try:
        weights = model.get_weights()
        serialized = pickle.dumps(weights)
            
        # Store with version
        version = mpredis.incr(f"{model_name}_version")
        mpredis.set(f"{model_name}_weights", serialized ,ex=60)  # expire in 60 seconds)
            
        print(f"Model saved to Redis - version {version}")
        return version
    except Exception as e:
        print(f"Error saving to Redis: {e}")
        return None
# def save_replay_memory():
#     """Save replay memory with compression"""
#     t1 = time.time()
#     try:
#         if len(replay_memory) == 0:
#             return
        
#         print(f"Saving replay memory ({len(replay_memory)} items)...")
#         with gzip.open(REPLAY_MEMORY_FILE, 'wb') as f:
#             pickle.dump(list(replay_memory), f, protocol=pickle.HIGHEST_PROTOCOL)
        
#         file_size = os.path.getsize(REPLAY_MEMORY_FILE) / (1024 * 1024)
#         print(f"✓ Saved {file_size:.2f} MB in {time.time() - t1:.2f} seconds")
#     except Exception as e:
#         print(f"✗ Error saving: {e}")

# def load_replay_memory():
#     """Load replay memory"""
#     # global replay_memory
#     t1 = time.time()
#     try:
#         if os.path.exists(REPLAY_MEMORY_FILE):
#             print(f"Loading replay memory...")
#             with gzip.open(REPLAY_MEMORY_FILE, 'rb') as f:
#                 data = pickle.load(f)
#             replay_memory.extend(data)
#             print(f"✓ Loaded {len(replay_memory)} items in {time.time() - t1:.2f} seconds")
#         else:
#             print("No saved memory found")
#     except Exception as e:
#         print(f"✗ Error loading: {e}")



REPLAY_MEMORY_KEY = "replay_memory_compressed"
REPLAY_MEMORY_VERSION_KEY = "replay_memory_version"

def save_replay_memory():
    """Save replay memory to Redis with compression"""
    t1 = time.time()
    try:
        if len(replay_memory) == 0:
            print("Replay memory is empty, nothing to save")
            return
        
        print(f"Saving replay memory ({len(replay_memory)} items) to Redis...")
        
        # Serialize
        serialized = pickle.dumps(list(replay_memory), protocol=pickle.HIGHEST_PROTOCOL)
        original_size = len(serialized) / (1024 * 1024)
        
        # Compress
        compressed = gzip.compress(serialized, compresslevel=6)
        compressed_size = len(compressed) / (1024 * 1024)
        
        # Save to Redis
        mpredis.set(REPLAY_MEMORY_KEY, compressed , ex=60*60*24)  # expire in 1 day
        
        # Increment version
        version = mpredis.incr(REPLAY_MEMORY_VERSION_KEY)
        
        compression_ratio = (1 - compressed_size/original_size) * 100 if original_size > 0 else 0
        print(f"✓ Saved {compressed_size:.2f} MB (compressed from {original_size:.2f} MB, {compression_ratio:.1f}% reduction)")
        print(f"  Version {version} in {time.time() - t1:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Error saving to Redis: {e}")
        import traceback
        traceback.print_exc()

def load_replay_memory():
    """Load replay memory from Redis with decompression"""
    t1 = time.time()
    try:
        if not mpredis.exists(REPLAY_MEMORY_KEY):
            print("No saved memory found in Redis")
            return
        
        print(f"Loading replay memory from Redis...")
        
        # Get compressed data from Redis
        compressed = mpredis.get(REPLAY_MEMORY_KEY)
        
        if compressed:
            compressed_size = len(compressed) / (1024 * 1024)
            
            # Decompress
            serialized = gzip.decompress(compressed)
            
            # Deserialize
            data = pickle.loads(serialized)
            replay_memory.extend(data)
            
            version = mpredis.get(REPLAY_MEMORY_VERSION_KEY)
            version = int(version) if version else 0
            
            print(f"✓ Loaded {len(replay_memory)} items from Redis")
            print(f"  Decompressed {compressed_size:.2f} MB, version {version} in {time.time() - t1:.2f} seconds")
        else:
            print("No data found in Redis")
            
    except Exception as e:
        print(f"✗ Error loading from Redis: {e}")
        import traceback
        traceback.print_exc()

def process_update_action(  actionId):
        # print('process_update_action called ' , p.reqIndex)
    # Recreate necessary agent if needed or call static function
        # return update_action(
        #     p.reqIndex,
        #     p.current_node_id,
        #     p.next_node_id,
        #     p.current_state,
        #     p.done_episode,
        #     p.timeSlot,
        #     p.reward,
        #     p.node_matrix,
        #     p.req_matrix,
        #     p.dist_matrix
        # )
        return update_action(actionId)

        # reqIndex , current_node_id,  action  , current_state  , done , timeSlot,lreward,ent_matrix , req_matrix,dist_matrix = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7].tolist(),p[8].tolist(),p[9].tolist()
        # return update_action(reqIndex , current_node_id,  action  , current_state  , done , timeSlot,lreward,ent_matrix , req_matrix,dist_matrix)

def update_action( actionId):
        # print('update_action called ')

        elem =  pickle.loads(mpredis.get(f"action_{actionId}"))
        mpredis.delete(f"action_{actionId}") 
        request_index , current_node_id,  action  , current_state  , done , timeSlot,lreward,ent_matrix , req_matrix,dist_matrix = elem[0],elem[1],elem[2],elem[3],elem[4],elem[5],elem[6],elem[7].tolist(),elem[8].tolist(),elem[9].tolist()

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
        del elem
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

        del req_tensor, attn_encoded, curr_emb, neighbor_embs, context_vec
        tf.keras.backend.clear_session()
        
        return ret

def process_actions(actionIds):
        # max_workers = 2
        # # global executor
        # if executor is None:
        #     executor = ProcessPoolExecutor(max_workers=max_workers)
        # self.executor
        print('process_actions called ' , len(actionIds), executor is not None)
        # futures = [executor.submit(self.process_update_action, p) for p in params]
        chunk_size = max(1, len(actionIds) // (max_workers))
        futures = list(executor.map(process_update_action, [p for p in actionIds] , chunksize=chunk_size))

        results = []
        # for f in as_completed(futures):
        #     results.append(f.result())
        for data in futures:
            try:
                results.append(data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing action: {e}")

        # Combine results
        actions = []
        for r in results:
            if r is not None:
                actions.append(r)


        return actions

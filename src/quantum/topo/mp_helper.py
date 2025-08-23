from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.managers import BaseManager
import os
import threading
import time
from .Node import sNode
from .Link import sLink
from multiprocessing import Lock
import logging, multiprocessing, os

logging.basicConfig(
    level=logging.INFO,
    format="%(processName)s [PID=%(process)d] %(message)s"
)

class qManager(BaseManager): pass
qManager.register('sNode', sNode,)
qManager.register('sLink', sLink)

# executor = ProcessPoolExecutor(max_workers=8)
executor = ThreadPoolExecutor(max_workers=128)

def route_schedule_single2(  reqState):
    print('route_schedule_single called with algo#############################################:')


def route_schedule_single(args):
        algo , nodes ,  reqState, lock = args  # Unpack the arguments

        """
        Serve only one request (reqState) using the routing agent.
        reqState: [src, dst, current_node, path, index, checked]
        """
        # topo = pickle.loads(serialized_topo)
        # nodes = pickle.loads(serialized_nodes)
        for _ in range(10):
            print('*****route_schedule_single called with algo:')
        # print(len(nodes))
        local_nodes = list(nodes)
        logging.info(f"Hello from task {[n.id for n in local_nodes]}")
        # algo = algo['algo']
        src, dst, current_node, path, index, checked = reqState
        current_node = nodes[current_node.id]  # Get the current node object
        selectedNodes = [src]
        selectedEdges = []
        selectedlinks = []
        usedLinks = []
        prev_links = None
        hopCount = 0
        success = False
        good_to_search = True
        failed_no_ent = False
        failed_loop = False
        failed_swap = False
        fail_hopcount = False
        path = list(path)
        numtry = 0
        maxTry = algo.maxTry
        fidelity = 1


        while good_to_search and not success and numtry <= maxTry:
            # Get next action for this request
            with lock:
                print('11111111111111')

                result = algo.routingAgent.learn_and_predict_next_req_node_single(reqState)

                print('22222222222222')

                if result is None:
                    break
                current_state, req_id, next_node_id, q, mask, valid_actions = result

                next_node = nodes[next_node_id]
                print('33333333333333', next_node.id)
                algo.tst.append(os.getpid())
                print('process id:', os.getpid(), 'thread id:', threading.get_ident(), 'next_node_id:', next_node_id , next_node , set(algo.tst))
                current_node_id = current_node.id
                # Find entangled links
                ent_links = [link for link in current_node.links if (link.isEntangled(algo.timeSlot) and link.contains(next_node) and link.notSwapped() and not link.taken)]
                print(f"Processing request {src.id} to {dst.id},current node ID: {current_node.id} next node ID: {next_node_id}", 'len ent_links:', len(ent_links) , 'path:', path)
                print('numtry:', numtry)
               
                # key = str(reqState[0].id) + '_' + str(reqState[1].id) + '_' + str(current_node.id) + '_' + str(next_node.id)
                print('numtry2:', numtry)
                if not ent_links:

                    numtry += 1
                    if numtry <= algo.maxTry:
                        continue
                    else:
                        good_to_search = False
                        failed_no_ent = True
                else:
                    ent_links = [ent_links[0]]
                    ent_links[0].taken = True
                    selectedlinks.append(ent_links[0])
                    prev_links = [ent_links[0]]
                    numtry = 0

                    # Fidelity check
                    fidelity = algo.fidelityAfterSwap(fidelity, ent_links[0].fidelity)
                    if fidelity < algo.topo.fidelity_threshold:
                        numtry += 1
                        if numtry <= algo.maxTry:
                                continue
                        else:
                            numtry = 0
                            good_to_search = False

                # print('44444444444444')
                print('next_node:', next_node.id)
                # Loop check
                if next_node == current_node or next_node.id in path:
                    good_to_search = False
                    failed_loop = True
                    # break
                print('55555555555555')
                selectedNodes.append(next_node)
                selectedEdges.append((current_node, next_node))
                # usedLinks.extend(prev_links)
                path.append(next_node.id)
                req_done = (not good_to_search) or success
                print('66666666666666')
                reqState = (src,dst,next_node,tuple(path),index,req_done)
                print('77777777777777')
                algo.requestState[index] = reqState

                # print('Updated request state:', reqState)
                # Success check
                if next_node == dst and good_to_search:
                    success = True
                    good_to_search = False

                # Prepare for next hop
                current_node = next_node
                swappSuccess = True
                if success:
                    print('going to swap for ' , (src.id , dst.id ))
                    for i in range(len(selectedlinks)-1):
                        l1 = selectedlinks[i]
                        l2 = selectedlinks[i+1]
                        n1 = l1.n1
                        n2 = l1.n2
                        n = n1 if l2.contains(n1) else n2
                        swapped = n.attemptSwapping(l1,l2)

                        usedLinks.append(l1)
                        usedLinks.append(l2)
                        if not swapped:
                            failed_swap = True
                            swappSuccess = False
                            print('================failed swap==================')
                            break
                


                if success and swappSuccess:
                    print('going to find path for:', (src.id , dst.id ))
                    t2 = time.time()
                    for req in algo.requests:
                            # src = req[0]
                            # dst = req[1]
                        if (src, dst) == (req[0], req[1]):
                                # print('[REPS] finish time:', algo.timeSlot - request[2])
                            algo.requests.remove(req)
                            break

                # successReq += 1
                # totalEntanglement += 1


            reward = -1

            if req_done:
                if success:
                        # print("====success====" , src.id , dst.id , [n for n in path])
                        # print('shortest path ----- ' , [n.id for n in targetPath])
                    reward = 10
                        # reward = 1

                    total_fidelity += fidelity
                else:
                    for link in selectedlinks:
                        link.taken = False
                    print("!!!!!!!=fail=!!!!!!!" , src.id , dst.id , [n for n in path] , 'threading.get_ident():', threading.get_ident())
                        # print('shortest path ----- ' , [n.id for n in targetPath])
                    
                    print('fail_hopcount' , fail_hopcount , 'failed_loop' , failed_loop , 'failed_no_ent' , failed_no_ent , 'failed_swap' , failed_swap)
                    reward = -10


                # print('lenT ' , len(T))

            # key = str(reqState[0].id) + '_' + str(reqState[1].id) + '_' + str(prev_node.id) + '_' + str(next_node.id)

                # reward = -algo.topo.numOfRequestPerRound

            try:
                algo.topo.reward_routing[key] += reward
            except:
                algo.topo.reward_routing[key] = reward    

            for link in usedLinks:
                link.clearPhase4Swap()
            
            T = [r for r in algo.requestState if not r[5]]
            done_episode = (not good_to_search or success) and (len(T)==1)
            
            algo.routingAgent.update_action( reqState ,current_node_id,  next_node_id  , current_state  , done_episode)
            

        return success and swappSuccess

# args = [i for i in range(10)]  # Example arguments for testing
#                 # print('args:', args)
#                 # with ProcessPoolExecutor(max_workers=8) as executor2:
#                 #
# results = list(executor.map(route_schedule_single2, args))

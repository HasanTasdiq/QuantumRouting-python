from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.managers import BaseManager
import os
import threading
import time
from .Node import sNode
from .Link import sLink
from multiprocessing import Lock
import logging, multiprocessing, os
import redis, dill
from multiprocessing import Lock, Manager

mpredis = redis.Redis()
lock1 = None
agent_lock = None
reward_lock = None
node_locks = {}


logging.basicConfig(
    level=logging.INFO,
    format="%(processName)s [PID=%(process)d] %(message)s"
)

class qManager(BaseManager): pass
qManager.register('sNode', sNode,)
qManager.register('sLink', sLink)

# executor = ProcessPoolExecutor(max_workers=8)
executor = ThreadPoolExecutor(max_workers=8)

def route_schedule_single2(  reqState):
    print('route_schedule_single called with algo#############################################:')


def getLock1():
    global lock1

    if lock1 is None:
        lock1 = Manager().Lock()
    return lock1

def update_shared_topo(task_id):
    print('update_shared_topo called with task_id:', task_id)
    with mpredis.pipeline() as pipe:
        while True:
            try:
                # Watch the key for changes
                pipe.watch("shared_topo")

                # Load the object
                obj = dill.loads(pipe.get("shared_topo"))

                # Modify it
                obj.tst.append(task_id)

                # Start transaction
                pipe.multi()
                pipe.set("shared_topo", dill.dumps(obj))
                pipe.execute()  # commits if key unchanged
                return obj
                break
            except redis.WatchError:
                # Retry if another process modified it concurrently
                continue
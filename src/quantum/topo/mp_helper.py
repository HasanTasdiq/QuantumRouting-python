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
mpredis = redis.Redis()

logging.basicConfig(
    level=logging.INFO,
    format="%(processName)s [PID=%(process)d] %(message)s"
)

class qManager(BaseManager): pass
qManager.register('sNode', sNode,)
qManager.register('sLink', sLink)

# executor = ProcessPoolExecutor(max_workers=32)
executor = ThreadPoolExecutor(max_workers=128)

def route_schedule_single2(  reqState):
    print('route_schedule_single called with algo#############################################:')




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
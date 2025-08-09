entanglement_lifetimeslot = 10
needlink_timeslot = 20
request_timeout = 5
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# executor = ProcessPoolExecutor(max_workers=8)
executor = ThreadPoolExecutor(max_workers=8)
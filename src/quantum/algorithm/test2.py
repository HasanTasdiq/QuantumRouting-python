import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time

def cpu_heavy_worker(args):
    shm_name, shape, dtype, row_idx = args
    shm = shared_memory.SharedMemory(name=shm_name)
    matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Heavy CPU task: repeated computations per row
    row = matrix[row_idx, :].copy()
    for _ in range(50):  # repeat to increase CPU load
        row = np.sin(row) ** 2 + np.cos(row) ** 2  # just a heavy trig operation
        row = row ** 3 + np.sqrt(row + 1)

    matrix[row_idx, :] = row  # write result back
    shm.close()
    return row_idx

if __name__ == "__main__":
    rows, cols = 40000, 1000  # large enough matrix
    matrix = np.random.rand(rows, cols)

    # Shared memory
    shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
    shared_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
    shared_matrix[:] = matrix[:]

    tasks = [(shm.name, matrix.shape, matrix.dtype, i) for i in range(rows)]

    num_cpus = mp.cpu_count()
    print(f"Using {num_cpus} CPU cores")

    start_time = time.time()

    with mp.Pool(processes=num_cpus) as pool:
        pool.map(cpu_heavy_worker, tasks)

    end_time = time.time()
    print("Total time taken for heavy CPU task: {:.2f} seconds".format(end_time - start_time))

    # Print a small slice to verify
    print("Modified matrix slice (first 5 rows):")
    print(shared_matrix[:5, :5])

    shm.close()
    shm.unlink()

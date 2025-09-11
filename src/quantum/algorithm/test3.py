import time
import math
from multiprocessing import cpu_count

def cpu_heavy_task(n: int) -> int:
    """Perform a CPU intensive task: count primes up to n."""
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return len(primes)

def benchmark(n: int = 200_0000):
    print(f"Running CPU benchmark (finding primes up to {n})")
    start = time.time()
    count = cpu_heavy_task(n)
    end = time.time()
    elapsed = end - start
    print(f"Primes found: {count}")
    print(f"Time taken: {elapsed:.4f} seconds")
    print(f"CPU cores available: {cpu_count()}")

if __name__ == "__main__":
    benchmark()

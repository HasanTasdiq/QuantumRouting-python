import psutil
import os

# Get the process ID of the current Python process
pid = os.getpid()

# Create a Process object using the process ID
process = psutil.Process(pid)

# Get the memory information for the process
memory_info = process.memory_info()

# Get the resident set size (RSS) in bytes, which represents the actual physical memory used
rss = memory_info.rss

# Convert the RSS to megabytes (MB)
rss_mb = rss / (1024 * 1024)

print(f"Memory usage: {rss_mb:.2f} MB")
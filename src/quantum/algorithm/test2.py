# quick_check_detailed.py
import psutil
import subprocess
import os
import re

def quick_memory_check():
    """Quick snapshot of memory state including shared memory details"""
    print("="*70)
    print("QUICK MEMORY CHECK")
    print("="*70)
    
    # Python processes
    python_procs = [p for p in psutil.process_iter(['pid', 'name', 'memory_info']) 
                    if 'python' in p.info['name'].lower()]
    
    total_python_mem = sum(p.info['memory_info'].rss for p in python_procs)
    print(f"\nPython Processes: {len(python_procs)}")
    print(f"Total Python Memory: {total_python_mem / 1024 / 1024:.2f} MB")
    
    for p in sorted(python_procs, key=lambda x: x.info['memory_info'].rss, reverse=True)[:5]:
        print(f"  PID {p.info['pid']:6d}: {p.info['memory_info'].rss / 1024 / 1024:8.2f} MB")
    
    # Shared memory - DETAILED
    print("\n" + "="*70)
    print("SHARED MEMORY SEGMENTS")
    print("="*70)
    
    try:
        result = subprocess.run(['ipcs', '-m'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        segments = []
        total_shm_size = 0
        orphaned_count = 0
        
        for line in lines:
            # Parse line: key shmid owner perms bytes nattch
            # Example: 0x00000000 163840 user 600 1048576 2
            match = re.search(r'(0x[0-9a-f]+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
            if match:
                key = match.group(1)
                shmid = match.group(2)
                owner = match.group(3)
                perms = match.group(4)
                size_bytes = int(match.group(5))
                nattch = int(match.group(6))
                
                segments.append({
                    'key': key,
                    'shmid': shmid,
                    'owner': owner,
                    'perms': perms,
                    'size': size_bytes,
                    'nattch': nattch
                })
                
                total_shm_size += size_bytes
                if nattch == 0:
                    orphaned_count += 1
        
        if segments:
            print(f"\nTotal Segments: {len(segments)}")
            print(f"Total Size: {total_shm_size / 1024 / 1024:.2f} MB ({total_shm_size:,} bytes)")
            print(f"Orphaned (0 attachments): {orphaned_count}")
            print("\nDetailed Breakdown:")
            print(f"{'ID':<12} {'Size (MB)':<12} {'Size (bytes)':<15} {'Attachments':<12} {'Status'}")
            print("-" * 70)
            
            # Sort by size descending
            segments.sort(key=lambda x: x['size'], reverse=True)
            
            for seg in segments:
                size_mb = seg['size'] / 1024 / 1024
                status = "⚠️ ORPHANED" if seg['nattch'] == 0 else "✓ Active"
                
                print(f"{seg['shmid']:<12} {size_mb:<12.2f} {seg['size']:<15,} {seg['nattch']:<12} {status}")
            
            # Show largest segments
            if len(segments) > 5:
                print(f"\nTop 5 Largest Segments:")
                for seg in segments[:5]:
                    print(f"  {seg['shmid']}: {seg['size'] / 1024 / 1024:.2f} MB")
            
            # Show orphaned segments
            orphaned = [s for s in segments if s['nattch'] == 0]
            if orphaned:
                print(f"\n⚠️  {len(orphaned)} Orphaned Segment(s) - Commands to remove:")
                for seg in orphaned:
                    print(f"  ipcrm -m {seg['shmid']}  # {seg['size'] / 1024 / 1024:.2f} MB")
        else:
            print("\n✓ No shared memory segments found")
            
    except Exception as e:
        print(f"\nError checking shared memory: {e}")
        print("Try running: ipcs -m")
    
    print("="*70)

if __name__ == "__main__":
    quick_memory_check()
import time
import sys
import numpy as np

def dp(N,A,B):
    R = 0.0;
    for j in range(0,N):
        R += A[j]*B[j]
    return R


assert len(sys.argv) == 3, "Incorrect input arguments"
N = np.uint64(sys.argv[1])
reps = np.uint64(sys.argv[2])
print(f'N {N} reps {reps}')
A = np.ones(N,dtype=np.float32)
B = np.ones(N,dtype=np.float32)
start_time = []
end_time = []
res = 0.0
for i in range(reps):
    start = time.monotonic_ns()
    res = dp(N, A, B)
    end = time.monotonic_ns()
    start_time.append(start)
    end_time.append(end)

print(res)
total_time = 0.0
for i in range(int(reps/2), reps):
    total_time += (end_time[i] - start_time[i])

avg_time = total_time/(reps/2)
bandwidth = N/(avg_time)
gflops = (2*N)/(avg_time)
avg_time_secs = avg_time/1000000000

print(f'N {N} <T>: {avg_time_secs:.6f} sec B: {bandwidth:.3f} GB/sec F:{gflops:.3f} GFLOPS/sec')

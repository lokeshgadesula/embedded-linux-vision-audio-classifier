import numpy as np
from edge_multi.core import benchmark
print(benchmark(lambda: np.fft.rfft(np.zeros(1024)),runs=100))

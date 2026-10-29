import numpy as np,time
def image(x):
 a=np.asarray(x,dtype=np.float32);a=a.mean(2) if a.ndim==3 else a;yi=np.linspace(0,a.shape[0]-1,64).astype(int);xi=np.linspace(0,a.shape[1]-1,64).astype(int);return (a[np.ix_(yi,xi)]/255)[None,...,None]
def spectrogram(w,n=128):
 a=np.asarray(w,dtype=np.float32);return np.log1p(np.abs(np.fft.rfft(a[:n]*np.hanning(n))))[None,:,None]
def fuse(v,a):p=(np.asarray(v)+np.asarray(a))/2;return int(np.argmax(p)),p
def benchmark(fn,*args,runs=5):
 x=[]
 for _ in range(runs):t=time.perf_counter();fn(*args);x.append((time.perf_counter()-t)*1000)
 x.sort();p=x[min(len(x)-1,int(.95*(len(x)-1)))];return {"p95_ms":p,"under_40ms":p<40}

import numpy as np
from edge_multi.core import *
def test_image():assert image(np.zeros((20,30,3),dtype=np.uint8)).shape==(1,64,64,1)
def test_audio():assert spectrogram(np.zeros(128)).shape==(1,65,1)
def test_fusion():assert fuse([.1,.9],[.2,.8])[0]==1
def test_benchmark():assert isinstance(benchmark(lambda:None,runs=2)["under_40ms"],bool)

import os

def enabled():
    return os.system("edmCUDAEnabled") == 0

def priority():
    return 2

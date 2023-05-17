import numpy as np
import pandas as pd
import os
import sys

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig
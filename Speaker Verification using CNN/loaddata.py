
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm


# In[5]:

from hw2.preprocess import *
from hw2.utils import *


# In[6]:

files = ['1','2','3','4','5','6','dev','test']
func = load(files)

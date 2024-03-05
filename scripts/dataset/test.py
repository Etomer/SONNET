import os,sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt

from src.simulation import simulate_room

sounds,toas = simulate_room(np.random.randn(30000))

print(toas)



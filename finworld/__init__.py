import numpy as np

# NumPy 2.x compatibility fix for Gym 0.26
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

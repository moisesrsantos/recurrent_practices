import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("stockmarket_f.data")


plt.figure()
plt.plot(df)
plt.show()

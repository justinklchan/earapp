import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set()

phone="iphone"
d0="/Users/justinkwoklamchan/data/"+phone+"/"
x=np.load(d0+"xall.npy")
y=np.load(d0+"yall.npy")

unique, counts = np.unique(y, return_counts=True)
counts=np.append([0],counts)

# np.random.shuffle(x)
numex=20
xrange=np.arange(2600)
colors=['r','b']

for i in range(len(counts)-1):
    seg=x[counts[i]:counts[i+1]]
    np.random.shuffle(seg)
    seg=seg[:numex]
    for j in seg:
        plt.plot(xrange,j,color=colors[i],linewidth=1)

plt.show()
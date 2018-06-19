import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
sns.set()
plt.figure()

phone="s6"
d0="/Users/justinkwoklamchan/data/"+phone+"/"

x=np.load(d0+"xall.npy")
y=np.load(d0+"yall.npy")

unique, counts = np.unique(y, return_counts=True)
counts = dict(zip(unique, counts))

labels=["pos-fft","neg-fft","err-fft"]
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=1000)
tsne_results = tsne.fit_transform(x)

prev=0
for i in counts:
    cat=tsne_results[prev:prev+counts[i]]
    sns.regplot(cat[:, 0], cat[:, 1],fit_reg=False,label=labels[i],scatter_kws={"s": 100})
    prev=counts[i]
plt.legend()

plt.show()
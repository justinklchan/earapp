import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(rc={"figure.figsize": (6, 6)})
np.random.seed(sum(map(ord, "palettes")))
sns.set_palette("husl")
current_palette = sns.color_palette()
# sns.palplot(current_palette)

plt.figure()
sns.regplot(np.asarray([1,2,3]),np.asarray([1,2,3]),fit_reg=False)

plt.show()
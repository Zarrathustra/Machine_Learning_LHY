# import pandas.rpy.common as com
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline

# iris = sns.load_dataset('iris')

data = pd.read_csv('data/covid.test.csv')
print(data)

# load the R package ISLR
# infert = com.importr("ISLR")

# load the Auto dataset
# auto_df = com.load_data('Auto')

# calculate the correlation matrix
corr = data.corr(method = 'pearson')
print(corr)
corr = corr.abs()

# plot the heatmap
sns.heatmap(corr, \
            xticklabels=corr.columns, \
            yticklabels=corr.columns)

plt.savefig('heatmap.png', dpi = 200)

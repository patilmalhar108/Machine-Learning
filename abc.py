import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
sns.set_style('dark')
print('Target variables:', dataset['target_names'])
(unique, counts) = np.unique(
dataset['target'],
return_counts=True
)
print('Unique values of the target variable:', unique)
print('Counts of the target variable:', counts)
sns.barplot(x=dataset['target_names'], y=counts)
plt.title('Target variable counts in dataset')
plt.show()
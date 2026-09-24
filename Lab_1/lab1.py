import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження датасету
iris = sns.load_dataset('iris')

class Zscore:
    def __init__(self, np_columns: np.ndarray):
        self.mean = np.nanmean(np_columns)
        self.std = np.nanstd(np_columns)
    
    def get_score(self, x):
        # Z = (x - μ) / σ
        return np.abs((x - self.mean) / self.std)
    


columns_to_analyze = ['sepal_length', 'sepal_width', 'petal_length']
all_z_scores = []

for col in columns_to_analyze:
        col_data = iris[col].to_numpy()
        score_calc = Zscore(col_data)
        scores = score_calc.get_score(col_data)
        all_z_scores.append(scores)


z_scores_matrix = np.array(all_z_scores).T

# агрегація
aggregated_z_scores = np.mean(z_scores_matrix, axis=1)

outliers_indices = np.where(aggregated_z_scores > 3)[0]

if len(outliers_indices) < 5:
    outliers_indices = np.argsort(aggregated_z_scores)[-5:]
    

print("Індекси викидів:", outliers_indices)


# ГРАФІК
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

all_indices = np.arange(len(iris))
normal_indices = np.setdiff1d(all_indices, outliers_indices)

# Норм дані
ax.scatter(iris.loc[normal_indices, columns_to_analyze[0]], 
           iris.loc[normal_indices, columns_to_analyze[1]], 
           iris.loc[normal_indices, columns_to_analyze[2]], 
           c='blue', label='Типові дані', alpha=0.5)

# Нетипові
ax.scatter(iris.loc[outliers_indices, columns_to_analyze[0]], 
           iris.loc[outliers_indices, columns_to_analyze[1]], 
           iris.loc[outliers_indices, columns_to_analyze[2]], 
           c='red', label='Outliers (Нетипові)', s=100, edgecolors='black')


ax.set_xlabel(columns_to_analyze[0])
ax.set_ylabel(columns_to_analyze[1])
ax.set_zlabel(columns_to_analyze[2])
ax.set_title("Візуалізація нетипових елементів (Z-score)")
plt.legend()


plt.show()
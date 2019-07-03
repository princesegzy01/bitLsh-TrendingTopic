import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
# box = dict(facecolor='yellow', pad=5, alpha=0.2)

# dataset = pd.read_csv('result/trigram.csv')
# # For line plotting
# ax = dataset.plot(x='dataset', y=['permutation_64', 'permutation_128', 'permutation_256',
#                                   'permutation_512'], style='', colormap='jet', lw=2, marker='.', markersize=10, title='TRIGRAM')
# ax.set(xlabel='Dataset', ylabel='Time (Minute)')
# ax.set_ylim(0, 200)
# plt.show()

dataset = pd.read_csv('result/all_512_perms.csv')
# for bar chart plotting                                                                                figsize = (15, 10), legend = True, fontsize = 12)
ax2 = dataset.plot.bar(
    x='dataset', y=['uni_perm_512', 'bi_perm_512', 'tri_perm_512'], title="ALL 512 PERMS", rot=0)
ax2.set_xlabel("Dataset", fontsize=12)
ax2.set_ylabel("Time (Minute)", fontsize=12)
ax2.set_ylim(0, 200)
plt.show()


# from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
# print("5000 Datasets Result")
# for all metrics
# df = pd.read_csv('result/5000_metrics.csv')
# y_true = df['y_true'].tolist()
# y_pred = df['y_pred'].tolist()


# print("Recall : ", recall_score(y_true, y_pred,  average='micro'))
# print("Presicion  : ", precision_score(y_true, y_pred, average='micro'))
# print("F1 Score : ", f1_score(y_true, y_pred, average='micro'))
# print("Accuracy : ", accuracy_score(y_true, y_pred))

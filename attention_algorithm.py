import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import os
# 环境变量
os.environ['OMP_NUM_THREADS'] = '1'

df = pd.read_csv('./data.csv')
data = df[['timestamp', 'left_x', 'left_y', 'right_x', 'right_y']].to_numpy()
maxarr = data.max(axis=0)
minarr = data.min(axis=0)


def split_by_timestamp(data, group_interval):
    current_group_start_time = data[0][0]
    current_group_end_time = current_group_start_time + group_interval
    current_group_data = []
    groups = []  # 存储所有的分组
    for datum in data:
        if datum[0] < current_group_end_time:
            # 如果时间戳在当前分组的时间范围内，将数据添加到当前分组
            current_group_data.append(datum)
        else:
            # 如果时间戳超出了当前分组的时间范围，将当前分组存储到groups列表中，并开始一个新的分组
            groups.append(current_group_data)
            current_group_data = [datum]
            current_group_start_time = current_group_end_time
            current_group_end_time = current_group_start_time + group_interval
    # 将最后一个分组存储到groups列表中
    groups.append(current_group_data)
    return groups


def clusting(data, axs):
    data = np.array(data)
    X = data[:, 1:]
    # n_clusters = int(len(X) / 10)
    n_clusters = 3
    if (len(X) < 3):
        return
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    # 使用KMeans算法进行聚类
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # print('Cluster centers:')
    # print(centers)
    axs.scatter(X[:, 0], X[:, 1], c=labels)
    axs.scatter(centers[:, 0],
                centers[:, 1],
                marker='x',
                s=200,
                linewidths=3,
                color='r')


group1 = split_by_timestamp(data, 3)

fig, axs = plt.subplots(int((len(group1[:10]) + 3 - 1) / 3),
                        3,
                        figsize=(10, 10))
for i in range(len(group1[:11])):
    clusting(group1[i], axs[int(i / 3)][i % 3])
plt.show()

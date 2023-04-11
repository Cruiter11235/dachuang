import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns

#读文件
def Read_data(file):
    eye_data = pd.read_csv(file)
    #去除第一行标签与第一列
    eye_data = eye_data.values[1:, 1:]

    #预处理
    eye_data = Preprocess(eye_data)

    #将str类型转化为float类型
    for i in range(len(eye_data)):
        eye_data[i] = list(map(float, eye_data[i]))

    #取出数据
    return eye_data

#数据预处理
def Preprocess(data):
    row = data.shape[0]
    arr = data.shape[1]

    for i in range(arr):
        if data[0][i] == 'None':
            data[0][i] = '0'
        for j in range(1, row):
            if data[j][i] == 'None':
                data[j][i] = data[j-1][i]

    return data

#画注视轨迹图, 在Jupyter notebook上运行不了
def draw_track(data):
    left_eye_x = data[:, 0]
    left_eye_y = data[:, 1]
    left_eye_d = data[:, 2]
    right_eye_x = data[:, 3]
    right_eye_y = data[:, 4]
    right_eye_d = data[:, 5]

    x_min = min(np.min(left_eye_x), np.min(right_eye_x))
    x_max = max(np.max(left_eye_x), np.max(right_eye_x))
    y_min = min(np.min(left_eye_y), np.min(right_eye_y))
    y_max = max(np.max(left_eye_y), np.max(right_eye_y))

    plt.ion()

    plt.subplots()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.rcParams["font.sans-serif"] = ['SimHei']
    for i in range(0, len(data)):
        plt.clf()
        plt.title("视线图(蓝色是左眼，红色是右眼，大小表示瞳孔大小)")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.scatter(left_eye_x[i], left_eye_y[i], color="blue", linewidths=left_eye_d[i])
        plt.scatter(right_eye_x[i], right_eye_y[i], color="red", linewidths=right_eye_d[i])
        #添加网络图
        plt.grid(True)
        plt.pause(0.0001)

    plt.ioff()
    plt.show()

#画热度图，感觉效果不好
def draw_heatmap(data, n=500):
    left_eye_x = data[:, 0]
    left_eye_y = data[:, 1]
    right_eye_x = data[:, 3]
    right_eye_y = data[:, 4]

    left_eye_x = (left_eye_x - np.min(left_eye_x)) / (np.max(left_eye_x) - np.min(left_eye_x))
    left_eye_x = list(map(int, n * left_eye_x))
    left_eye_y = (left_eye_y - np.min(left_eye_y)) / (np.max(left_eye_y) - np.min(left_eye_y))
    left_eye_y = list(map(int, n * left_eye_y))
    right_eye_x = (right_eye_x-np.min(right_eye_x)) / (np.max(right_eye_x)-np.min(right_eye_x))
    right_eye_x = list(map(int, n*right_eye_x))
    right_eye_y = (right_eye_y-np.min(right_eye_y)) / (np.max(right_eye_y)-np.min(right_eye_y))
    right_eye_y = list(map(int, right_eye_y))

    data_length = len(data)
    heatmap = np.zeros((n + 1, n + 1))
    for i in range(data_length):
        heatmap[left_eye_x[i]][left_eye_y[i]] += 1
        heatmap[right_eye_x[i]][right_eye_y[i]] += 1

    ax = plt.subplots()
    ax = sns.heatmap(heatmap, cmap="YlGnBu", vmax=8, vmin=0)
    plt.show()


def draw_scatter(data):
    left_eye_x = data[:, 0]
    left_eye_y = data[:, 1]
    right_eye_x = data[:, 3]
    right_eye_y = data[:, 4]

    plt.figure()
    plt.scatter(left_eye_x, left_eye_y, alpha=0.01)
    plt.scatter(right_eye_x, right_eye_y, alpha=0.01)

    plt.show()

#main
eye_data = Read_data('./out.csv')
# draw_track(eye_data)
draw_heatmap(eye_data)
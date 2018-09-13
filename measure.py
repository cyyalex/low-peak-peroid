#城市坐标转换成欧氏距离
import numpy as np
def getdistmat(coordinates,numcity):
    num = coordinates.shape[0]
    distmat = np.zeros((numcity, numcity))
    # 初始化生成52*52的矩阵
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = round(np.linalg.norm(coordinates[i] - coordinates[j]),1)
    return distmat

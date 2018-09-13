# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: wsw

import numpy as np
from aco import measure
import random
import matplotlib.pyplot as plt


generations=2#遗传迭代次数
pc=0.88 #交配概率
pm=0.5 #变异概率
fitness =[] #适应度
population =[] #种群对应的十进制数值
population_size=10 #种群数量
fitness_sum=[]
optimum_solution=[] #每次迭代所获得的最优解
best_solution=[]   #最终结果
population_proportion=[]#每个染色体适应度总和的比
numant = 10  # 蚂蚁个数
shortest = []
shortestlength=[]
population_alllist = []

# shape[0]=52 城市个数,也就是任务个数
Q = 1  # 完成率
iter = 0  # 迭代初始
itermax = 10  # 迭代总数


#生成初始种群 α∈（0,3） β∈（2,5） ρ∈（0,0.5）
def init_population():
    N = np.random.uniform( size=(population_size, 4))   #做一个5X4的随机矩阵
    for i in range(population_size):
        N[i][0] = i + 1   #序号列
        N[i][1] = round(np.random.uniform(0,3),1)       #α∈（0,3）
        N[i][2] = round(np.random.uniform(2, 5),1)      #β∈（2，5）
        N[i][3] = round(np.random.uniform(0, 0.5),1)        #ρ∈（0,0.5）
    population.extend(N)
    return population  # 信息素重要程度因子# 启发函数重要程度因子# 信息素的挥发速度




#将参数放入蚁群中
def aco():
    #iter1=0           #循环populations_size*itermax次
    global alpha,beta,rho,lengthbest,lengthaver,pathbest,numcity,coordinates
    coordinates = np.array([[37.0, 52.0], [49.0, 49.0], [52.0, 64.0], [20.0, 26.0], [40.0, 30.0],
                            [21.0, 47.0], [17.0, 63.0], [31.0, 62.0], [52.0, 33.0], [51.0, 21.0],
                            [42.0, 41.0], [31.0, 32.0], [5.0, 25.0], [12.0, 42.0], [36.0, 16.0],
                            [52.0, 41.0], [27.0, 23.0], [17.0, 33.0], [13.0, 13.0], [57.0, 58.0],
                            [62.0, 42.0], [42.0, 57.0], [16.0, 57.0], [8.0, 52.0], [7.0, 38.0],
                            [27.0, 68.0], [30.0, 48.0], [43.0, 67.0], [58.0, 48.0], [58.0, 27.0],
                            [37.0, 69.0], [38.0, 46.0], [46.0, 10.0], [61.0, 33.0], [62.0, 63.0],
                            [63.0, 69.0], [32.0, 22.0], [45.0, 35.0], [59.0, 15.0], [5.0, 6.0],
                            [10.0, 17.0], [21.0, 10.0], [5.0, 64.0], [30.0, 15.0], [39.0, 10.0],
                            [32.0, 39.0], [25.0, 32.0], [25.0, 55.0], [48.0, 28.0], [56.0, 37.0],
                            [30.0, 40.0]])

    # 返回城市距离矩阵
    numcity = coordinates.shape[0]
    distmat = measure.getdistmat(coordinates,numcity)
    # print(distmat)   #输出欧氏距离矩阵


    etatable = 1.0 / (distmat + np.diag([1e10] * numcity))
    # diag(),将一维数组转化为方阵 启发函数矩阵，表示蚂蚁从城市i转移到城市j的期望程度
    pheromonetable = np.ones((numcity, numcity))
    # 信息素矩阵 52*52
    pathtable = np.zeros((numant, numcity)).astype(int)
    # 路径记录表，转化成整型 40*52
    distmat = measure.getdistmat(coordinates,numcity)
    # 城市的距离矩阵 52*52

    lengthaver = np.zeros(itermax)  # 迭代50次，存放每次迭代后，路径的平均长度  50*1
    lengthbest = np.zeros(itermax)  # 迭代50次，存放每次迭代后，最佳路径长度  50*1
    pathbest = np.zeros((itermax, numcity))  # 迭代50次，存放每次迭代后，最佳路径城市的坐标 50*52
    for q in range(population_size):  #五组参数在蚁群算法中循环
        alpha=population[q][1]
        beta=population[q][2]
        rho=population[q][3]
        for iter in range(itermax):
            # 迭代总数

            # 40个蚂蚁随机放置于52个城市中
            if numant <= numcity:  # 城市数比蚂蚁数多，不用管
                pathtable[:, 0] = np.random.permutation(range(numcity))[:numant]
                # 返回一个打乱的40*52矩阵，但是并不改变原来的数组,把这个数组的第一列(40个元素)放到路径表的第一列中
                # 矩阵的意思是哪个蚂蚁在哪个城市,矩阵元素不大于52
            else:  # 蚂蚁数比城市数多，需要有城市放多个蚂蚁
                pathtable[:numcity, 0] = np.random.permutation(range(numcity))[:]
                # 先放52个
                pathtable[numcity:, 0] = np.random.permutation(range(numcity))[:numant - numcity]
                # 再把剩下的放完
            #print(pathtable[:,0])
            length = np.zeros(numant)  # 1*40的数组

            # 本段程序算出每只/第i只蚂蚁转移到下一个城市的概率
            for i in range(numant):

                # i=0
                visiting = pathtable[i, 0]  # 当前所在的城市
                # set()创建一个无序不重复元素集合
                #visited = set() #已访问过的城市，防止重复
                #visited.add(visiting) #增加元素
                #print(visited)
                unvisited = set(range(numcity))
                # 未访问的城市集合
                # 剔除重复的元素
                unvisited.remove(visiting)  # 删除已经访问过的城市元素

                for j in range(1, numcity):  # 循环numcity-1次，访问剩余的所有numcity-1个城市
                    # j=1
                    # 每次用轮盘法选择下一个要访问的城市
                    listunvisited = list(unvisited)
                    # 未访问城市数,list
                    probtrans = np.zeros(len(listunvisited))
                    # 每次循环都初始化转移概率矩阵1*52,1*51,1*50,1*49....

                    # 以下是计算转移概率
                    for k in range(len(listunvisited)):
                        probtrans[k] = (pheromonetable[visiting][listunvisited[k]]**alpha) \
                                       * (etatable[visiting][listunvisited[k]]**alpha)
                    #eta-从城市i到城市j的启发因子 这是概率公式的分母   其中[visiting][listunvis[k]]是从本城市到k城市的信息素
                    cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
                    # 求出本只蚂蚁的转移到各个城市的概率斐波衲挈数列

                    cumsumprobtrans -= np.random.rand()
                    # 随机生成下个城市的转移概率，再用区间比较
                    #k = listunvisited[ndarray.find(cumsumprobtrans > 0)[0]]
                    k = listunvisited[list(cumsumprobtrans > 0).index(True)]
                    #k = listunvisited[np.where(cumsumprobtrans > 0)[0]]
                    # where 函数选出符合cumsumprobtans>0的数
                    # 下一个要访问的城市

                    pathtable[i, j] = k
                    #print(pathtable)
                    # 采用禁忌表来记录蚂蚁i当前走过的第j城市的坐标，这里走了第j个城市.k是中间值
                    unvisited.remove(k)
                    # visited.add(k)
                    # 将未访问城市列表中的K城市删去，增加到已访问城市列表中

                    length[i] += distmat[visiting][k]
                    # 计算本城市到K城市的距离
                    visiting = k

                length[i] += distmat[visiting][pathtable[i, 0]]
                # 计算本只蚂蚁的总的路径距离，包括最后一个城市和第一个城市的距离

            #print("ants all length:",length)
            # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数（每只蚂蚁遍历完城市后总路程）

            lengthaver[iter] = length.mean()
            # 本轮的平均路径

            # 本部分是为了求出最佳路径

            if iter == 0:
                lengthbest[iter] = length.min()
                pathbest[iter] = pathtable[length.argmin()].copy()
            # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
            else:
                # 后面几轮的情况，更新最佳路径
                if length.min() > lengthbest[iter - 1]:
                    lengthbest[iter] = lengthbest[iter - 1]
                    pathbest[iter] = pathbest[iter - 1].copy()
                # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
                else:
                    lengthbest[iter] = length.min()
                    pathbest[iter] = pathtable[length.argmin()].copy()
            # 此部分是为了更新信息素
            changepheromonetable = np.zeros((numcity, numcity))
            for i in range(numant):  # 更新所有的蚂蚁
                for j in range(numcity - 1):
                    changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][
                        pathtable[i, j + 1]]
                    # 根据公式更新本只蚂蚁改变的城市间的信息素      Q/d   其中d是从第j个城市到第j+1个城市的距离
                changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / distmat[pathtable[i, j + 1]][pathtable[i, 0]]
                # 首城市到最后一个城市 所有蚂蚁改变的信息素总和

            # 信息素更新公式p=(1-挥发速率)*现有信息素+改变的信息素
            pheromonetable = (1 - rho) * pheromonetable + changepheromonetable

            iter += 1  # 迭代次数指示器+1

            #print("this iteration end：", iter)
            # 观察程序执行进度，该功能是非必须的

            #if (iter - 1) % 20 == 0:
                #print("schedule:",iter - 1)
        # 迭代完成

        path=tuple(pathbest[len(pathbest)-1])

        shortlength=lengthbest[iter-1]#取出最短路径
        #print('当前参数最短路径:',shortlength)          #输出每组参数的最短路径
        shortest.append(shortlength)

        #np.delete(lengthbest,0,axis=0)
        population_eachlist = []
        population_eachlist.append(alpha)
        population_eachlist.append(beta)
        population_eachlist.append(rho)
        population_eachlist.append(shortlength)
        population_eachlist.append(path)
        population_alllist.append(population_eachlist)
    #print('返回所有参数组合所计算的最短路径', shortest)
    #print("种群表", population_alllist)  # 制作一个种群表[first[alpha,beta,rho,lengthbest,[pathbest]],second[...]...]避免对应错误



#适应值评价
def calculate_fitness():
    for i in range(population_size):
        function_value=1/population_alllist[i][3]
        fitness.append(function_value)

    #print('每个种群的适应值：',fitness)




#获取最大适应度的个体(每轮遗传最优)
def best_value():
    best_fitness=population_alllist[0]
    min_fitness=population_alllist[0][3]
    for i in range(population_size):
        if population_alllist[i][3]<min_fitness:
            min_fitness=population_alllist[i][3]
            best_fitness=population_alllist[i]
    optimum_solution.append(best_fitness)
    #print('遗传迭代每轮最优种群表：',optimum_solution)
    return optimum_solution

#获取所有遗传最优
def best_ga():
    ga_best=optimum_solution[0]
    ga_good=optimum_solution[0][3]
    for i in range(generations):
        if optimum_solution[i][3]<ga_good:
            ga_good=optimum_solution[i][3]
            ga_best=optimum_solution[i]
    best_solution.extend(ga_best)
    #print("最优种群表",best_solution)
    return best_solution



#采用轮盘赌算法进行选择过程
def selection():
    fitness_sum = 0
    for i in range(population_size):
        fitness_sum += fitness[i]
        # 计算生存率
    for i in range(population_size):
        population_proportion.append(fitness[i] / fitness_sum)
    pie_fitness = []
    cumsum = 0.0
    for i in range(population_size):
        pie_fitness.append(cumsum + population_proportion[i])
        cumsum += population_proportion[i]
    # 生成随机数在轮盘上选点[0, 1)
    random_selection = []
    for i in range(population_size):
        random_selection.append(random.random())
    # 选择新种群
    new_population = []
    random_selection_id = 0
    global population
    for i in range(population_size):
        while random_selection_id < population_size and random_selection[random_selection_id] < pie_fitness[i]:
            new_population.append(population[i])
            random_selection_id += 1
    population = new_population
    #print(population)          #输出新种群


#进行交配
def crossover():
    for i in range(0,population_size-1,2):
        if random.random()<pc:
            #随机选择交叉点
            change_point=random.randint(1,3)
            temp1=[]
            temp2=[]
            temp1.extend(population[i][0:change_point])
            temp1.extend(population[i+1][change_point:])
            temp2.extend(population[i+1][0:change_point])
            temp2.extend(population[i][change_point:])
            population[i]=temp1
            population[i+1]=temp2
    #print(population[i])
    #print(population[i+1])

def mutation():
    for i in range(population_size):
        if random.random()<pm:
            mutation_point=random.randint(1,3)#随机变异点
            if mutation_point==1:      #如果α变异
                population[i][mutation_point]=round(random.uniform(0,3),1)
            else:
                if mutation_point==2:       #如果β变异
                    population[i][mutation_point] = round(random.uniform(2,5),1)
                else:                   #如果ρ变异
                    population[i][mutation_point]=round(random.uniform(0,0.5),1)


# 以下是做图部分
# 做出平均路径长度和最优路径长度
def pictrue():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    axes[0].plot(lengthaver, 'k', marker='*')
    axes[0].set_title('Average Length')
    axes[0].set_xlabel(u'iteration')

    # 线条颜色black https://blog.csdn.net/ywjun0919/article/details/8692018
    axes[1].plot(lengthbest, 'k', marker='<')
    axes[1].set_title('Best Length')
    axes[1].set_xlabel(u'iteration')
    fig.savefig('Average_Best.png', dpi=500, bbox_inches='tight')
    plt.close()
    #fig.show()

    # 作出找到的最优路径图
    bestpath =best_solution[4]

    plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker='>')
    plt.xlim([0, 100])
    # x范围
    plt.ylim([0, 100])
    # y范围

    for i in range(numcity - 1):
        # 按坐标绘出最佳两两城市间路径
        m, n = int(bestpath[i]), int(bestpath[i + 1])
        print("best_path:", m, n)
        plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')

    plt.plot([coordinates[int(bestpath[0])][0], coordinates[int(bestpath[28])][0]],
             [coordinates[int(bestpath[0])][1], coordinates[int(bestpath[27])][1]], 'b')

    ax = plt.gca()
    ax.set_title("Best Path")
    ax.set_xlabel('X_axis')
    ax.set_ylabel('Y_axis')

    plt.savefig('Best Path.png', dpi=500, bbox_inches='tight')
    plt.show()


init_population()
for step in range(generations):
    aco()
    calculate_fitness()
    shortest.clear()
    best_value()
    selection()
    fitness.clear()
    crossover()
    mutation()
    population_alllist.clear()

best_ga()


print('alpha:',best_solution[0])
print('beta:',best_solution[1])
print('rho:',best_solution[2])
print('最短路径:',best_solution[3])
pictrue()

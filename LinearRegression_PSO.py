
# y = ax + b로의 linear regression
#
# xi, yi가 주어졌을 때 손실함수 C(a, b) = (yi - (a*xi + b))^2 이며, 모든 데이터의 손실함수의 평균이 현재의 손실함수 값이다.
# 손실함수를 최소화시키기 위하여, PSO(Particle Swarm Optimization)를 사용하였다.
# 시간 t에서 각 particle 들의 위치를 S(t), 방향벡터를 V(t)라고 하면,
# V(t) = w*V(t-1) + c1*r1*(personalBest(t) - S(t)) + c2*r2*(globalBest(t) - S(t))
# S(t+1) = S(t) + V(t)


import random


# 데이터 세트
dataSet_input = [5, -15, 7, -7, -11, 3, -9, 1, -1, 11, 9, -13, 13, -3, -5]
dataSet_output = [18, -82, 28, -42, -62, 8, -52, -2, -12, 48, 38, -72, 58, -22, -32]

newDataSet = [-11, 4, -9, 2, -10, -2, 0, -3, 1, -6, -7, -5, -8, -1]


class Particle:
    globalBest = [0, 0]
    globalBestCost = 0
    # c1, c2는 학습요인의 가중치이며, w는 방향벡터의 관성가중치
    c1 = 1
    c2 = 1
    w = 0.5

    # 임의의 값으로 각 particle 의 위치벡터/방향벡터 초기화
    def __init__(self):
        self.location = [random.random() * 100, random.random() * 100]
        self.direction = [random.random() * 100, random.random() * 100]
        self.personalBest = self.location
        self.personalBestCost = self.calculateCost()

    # Particle 의 이번 위치벡터/방향벡터 계산
    def move(self):
        self.direction = list(map(lambda x:
                              self.w * x[1]
                              + Particle.c1 * random.random() * (self.personalBest[x[0]] - self.location[x[0]])
                              + Particle.c2 * random.random() * (Particle.globalBest[x[0]] - self.location[x[0]])
                              , enumerate(self.direction)))
        self.location = list(map(lambda x, y: x + y, self.location, self.direction))

    # 현재의 손실함수 값과 자신의 최저손실함수 값, 전역최저손실함수 값을 비교하여 업데이트
    def updateBest(self):
        currentCost = self.calculateCost()
        if self.personalBestCost > currentCost:
            self.personalBest = self.location
            self.personalBestCost = currentCost
        if Particle.globalBestCost > currentCost:
            Particle.globalBest = self.location
            Particle.globalBestCost = currentCost

    # Optimization 시작 전 globalBest 값 초기화
    def initializeGlobalBest(self):
        Particle.globalBest = self.personalBest
        Particle.globalBestCost = self.personalBestCost

    # 손실함수 값 계산
    def calculateCost(self):
        cost_average = 0
        for i in range(len(dataSet_input)):
            result = self.location[0] * dataSet_input[i] + self.location[1]
            cost = pow(result - dataSet_output[i], 2)
            cost_average = calculateAverage(cost_average, i + 1, cost)
        return cost_average


# 재귀식 평균계산 함수
def calculateAverage(lastAverage, n, an):
    return lastAverage * (n - 1) / n + an / n


if __name__ == "__main__":
    # Particle 개체 수 선정 및 particle initialization
    particleNum = 100
    particleList = []
    for _ in range(particleNum):
        particleList.append(Particle())
    particleList[0].initializeGlobalBest()

    # 주어진 iteration 동안 particle 들이 이동 및 최저손실함수 업데이트
    iteration = 500
    for simulation in range(iteration):
        for particle in particleList:
            particle.move()
            particle.updateBest()
        print("Iteration #%d - Global lowest cost: %.20f" % (simulation, Particle.globalBestCost))

    # Particle 들의 평균 최종위치 계산
    averageLocation = [0, 0]
    for index, particle in enumerate(particleList):
        averageLocation = list(map(lambda x: calculateAverage(x[1], index + 1, particle.location[x[0]])
                                   , enumerate(averageLocation)))
        print("Particle #%d location:" % index, particle.location)
    print("Particle average location:", averageLocation)

    # 학습된 a, b를 통해 새로운 데이터 세트로 부터 예상 결과 출력
    for data in newDataSet:
        print("Input data: %d, Estimated output: %d" % (data, averageLocation[0] * data + averageLocation[1]))

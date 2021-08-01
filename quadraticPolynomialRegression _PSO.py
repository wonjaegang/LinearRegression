
# Q) 2차 함수로 회귀
#
# xi, yi가 주어졌을 때 손실함수 C(a, b) = (target - result)^2 이며, 모든 데이터의 손실함수의 평균이 현재의 손실함수 값이다.
# 손실함수를 최소화시키기 위하여, PSO(Particle Swarm Optimization)를 사용하였다.
# 시간 t에서 각 particle 들의 위치를 S(t), 방향벡터를 V(t)라고 하면,
# V(t) = w*V(t-1) + c1*r1*(personalBest(t) - S(t)) + c2*r2*(globalBest(t) - S(t))
# S(t+1) = S(t) + V(t)


import random
import pygame
pygame.init()


# 데이터 세트
dataSet_input = [0, 1, 3, 4, 5, 6, -1, 11, 9, -13, 13, -3, -5]
dataSet_output = [5, 16, 44, 61, 80, 101, -4, 236, 176, 44, 304, -16, -20]

newDataSet = [-11, 4, -9, 2, -10, -2, 0, -3, 1, -6, -7, -5, -8, -1]

# GUI 관련 변수
SCREEN_SIZE = 500
screen = pygame.display.set_mode((SCREEN_SIZE * 3, SCREEN_SIZE))
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

CELL_SIZE = 50
COLUMN_COUNT = 30
ROW_COUNT = 10
particleSize = 3


# GUI: 격자 및 텍스트 출력 함수
def printGrid():
    for column_index in range(COLUMN_COUNT):
        for row_index in range(ROW_COUNT):
            gridRect = (CELL_SIZE * column_index, CELL_SIZE * row_index, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, gridRect)
            pygame.draw.rect(screen, BLACK, gridRect, 1)

    font = pygame.font.SysFont('arial', 20, True, False)
    gains = font.render("W = %.3f     C1 = %.3f     C2 = %.3f" % (Particle.w, Particle.c1, Particle.c2), True, BLACK)
    globalBest = font.render("Global lowest loss function: %.20f" % Particle.globalBestCost, True, BLACK)
    x1 = font.render("→ x", True, BLACK)
    x2 = font.render("↑ x", True, BLACK)
    y1 = font.render("→ y", True, BLACK)
    y2 = font.render("↑ y", True, BLACK)
    z1 = font.render("→ z", True, BLACK)
    z2 = font.render("↑ z", True, BLACK)
    screen.blit(gains, (1000, 440))
    screen.blit(globalBest, (1000, 465))
    screen.blit(x1, (460, 225))
    screen.blit(y1, (460 + SCREEN_SIZE, 225))
    screen.blit(z1, (460 + SCREEN_SIZE * 2, 225))
    screen.blit(y2, (255, 5))
    screen.blit(z2, (255 + SCREEN_SIZE, 5))
    screen.blit(x2, (255 + SCREEN_SIZE * 2, 5))


class Particle(pygame.Rect):
    globalBest = [0, 0, 0]
    globalBestCost = 0
    # c1, c2는 학습요인의 가중치이며, w는 방향벡터의 관성가중치
    c1 = 1
    c2 = 1
    w = 0.6

    # 임의의 값으로 각 particle 의 위치벡터/방향벡터 초기화
    def __init__(self):
        self.location = [random.uniform(-1, 1) * 500, random.uniform(-1, 1) * 500, random.uniform(-1, 1) * 500]
        self.direction = [random.uniform(-1, 1) * 500, random.uniform(-1, 1) * 500, random.uniform(-1, 1) * 500]
        self.personalBest = self.location
        self.personalBestCost = self.calculateCost()

        # GUI: rect 객체 초기화
        super().__init__(list(map(lambda x: x, self.location[:2])), [particleSize, particleSize])

    # Particle 의 이번 위치벡터/방향벡터 계산
    def moveParticle(self):
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
            result = self.location[0] * dataSet_input[i] * dataSet_input[i] +\
                     self.location[1] * dataSet_input[i] +\
                     self.location[2]
            cost = pow(result - dataSet_output[i], 2)
            cost_average = calculateAverage(cost_average, i + 1, cost)
        return cost_average

    def drawParticle(self):
        self.left = self.location[0] + SCREEN_SIZE / 2
        self.top = SCREEN_SIZE / 2 - self.location[1]
        pygame.draw.rect(screen, RED, self)

        self.left = self.location[1] + SCREEN_SIZE / 2 + SCREEN_SIZE
        self.top = SCREEN_SIZE / 2 - self.location[2]
        pygame.draw.rect(screen, GREEN, self)

        self.left = self.location[2] + SCREEN_SIZE / 2 + SCREEN_SIZE * 2
        self.top = SCREEN_SIZE / 2 - self.location[0]
        pygame.draw.rect(screen, BLUE, self)


# 재귀식 평균계산 함수
def calculateAverage(lastAverage, n, an):
    return lastAverage * (n - 1) / n + an / n


if __name__ == "__main__":
    # Particle 개체 수 선정 및 particle initialization
    particleNum = 1000
    particleList = []
    for _ in range(particleNum):
        particleList.append(Particle())
    particleList[0].initializeGlobalBest()

    # 주어진 iteration 동안 particle 들이 이동 및 최저손실함수 업데이트
    # GUI: iteration 마다 particle 위치 업데이트 후 출력
    iteration = 200
    for simulation in range(iteration):
        printGrid()
        for particle in particleList:
            particle.moveParticle()
            particle.updateBest()
            particle.drawParticle()
        print("Iteration #%d - Global lowest cost: %.20f" % (simulation, Particle.globalBestCost))
        pygame.display.update()
        pygame.time.wait(50)

    # Particle 들의 평균 최종위치 계산
    averageLocation = [0 for _ in particleList[0].location]
    for index, particle in enumerate(particleList):
        averageLocation = list(map(lambda x: calculateAverage(x[1], index + 1, particle.location[x[0]])
                                   , enumerate(averageLocation)))
        print("Particle #%d location:" % index, particle.location)
    print("Particle average location:", averageLocation)

    # 학습된 a, b를 통해 새로운 데이터 세트로 부터 예상 결과 출력
    for data in newDataSet:
        print("Input data: %d, Estimated output: %d" % (data, averageLocation[0] * data + averageLocation[1]))

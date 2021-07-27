import pygame
import random
pygame.init()


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

CELL_SIZE = 100
COLUMN_COUNT = 10
ROW_COUNT = 10


def printGrid():
    # 격자 생성
    for column_index in range(COLUMN_COUNT):
        for row_index in range(ROW_COUNT):
            gridRect = (CELL_SIZE * column_index, CELL_SIZE * row_index, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, gridRect)
            pygame.draw.rect(screen, BLACK, gridRect, 1)


while True:
    screen.fill(WHITE)
    printGrid()

    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        break

    # particle initialize
    rectList = []
    for i in range(100):
        location = [i * 10, i * 10]
        rectList.append(pygame.Rect(location, [10, 10]))
    for rect in rectList:
        pygame.draw.rect(screen, RED, rect)
        pygame.display.update()

    for _ in range(500):
        printGrid()
        for rect in rectList:
            rect.left = rect.left + random.random() * 5
            rect.top = rect.top + random.random() * 5
            pygame.draw.rect(screen, RED, rect)
        pygame.display.update()
        pygame.time.wait(10)

    clock.tick(30)
    break

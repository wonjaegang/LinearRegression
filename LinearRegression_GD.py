
# y = ax + b로의 linear regression
#
# xi, yi가 주어졌을 때 손실함수 C(a, b) = (yi - (a*xi + b))^2 이며, 모든 데이터의 손실함수의 평균이 현재의 손실함수 값이다.
# 손실함수를 최소화시키기 위하여, 경사구배법을 사용하였다.
# 즉, 손실함수의 그래디언트를 구하고, a, b를 그래디언트의 반대방향으로 업데이트 해나가며 손실함수를 최소화시키는 a, b를 찾는다.
# 손실함수의 a, b에 대한 그래디언트는 손실함수를 a, b로 편미분하여 얻을 수 있다.
# 즉, 데이터 xi, yi 에서의 Gradient(C) = (∂c/∂a, ∂c/∂b) = (-2xi(yi - (a*xi + b)), -2(yi - (a*xi + b)))
# 전체 데이터 셋의 그래디언트는 각 데이터의 그래디언트를 평균을 취하면 얻을 수 있다.


# 데이터 세트
dataSet_input = [5, -15, 7, -7, -11, 3, -9, 1, -1, 11, 9, -13, 13, -3, -5]
dataSet_output = [18, -82, 28, -42, -62, 8, -52, -2, -12, 48, 38, -72, 58, -22, -32]

newDataSet = [-11, 4, -9, 2, -10, -2, 0, -3, 1, -6, -7, -5, -8, -1]


# 재귀식 평균계산 함수
def calculateAverage(lastAverage, n, an):
    return lastAverage * (n - 1) / n + an / n


if __name__ == "__main__":
    # 임의의 값으로 a, b를 초기화
    a = 0
    b = 0

    # epoch : 데이터 세트 반복 학습 횟수. 튜닝결과, 1000번이면 정확한 a, b를 충분히 찾음
    epoch = 1000
    for epo in range(epoch):
        dC_da_average = 0
        dC_db_average = 0
        # 각 데이터에 대해 그래디언트 값을 구하고 평균을 취해 전체 데이처 셋의 그래디언트 계산
        for i in range(len(dataSet_input)):
            dC_da = -2 * dataSet_input[i] * (dataSet_output[i] - (a * dataSet_input[i] + b))
            dC_db = -2 * (dataSet_output[i] - (a * dataSet_input[i] + b))
            dC_da_average = calculateAverage(dC_da_average, i + 1, dC_da)
            dC_db_average = calculateAverage(dC_db_average, i + 1, dC_db)

        # 그래디언트의 반대방향으로 a, b를 업데이트
        # learningRate 는 학습률로 얼마나 빠르게 학습할 지를 결정. 튜닝결과, 0,01보다 클 경우 overshooting 이 발생
        learningRate = 0.01
        a = a - dC_da_average * learningRate
        b = b - dC_db_average * learningRate

        # 현재 학습된 a, b와 그 손실함수를 계산 후 출력
        cost_average = 0
        for i in range(len(dataSet_input)):
            result = a * dataSet_input[i] + b
            cost = pow(result - dataSet_output[i], 2)
            cost_average = calculateAverage(cost_average, i + 1, cost)
        print("a: %f, b: %f" % (a, b))
        print("Epoch: %d, Current loss function: %.20f" % (epo, cost_average))
        print("=" * 60)

    # 학습된 a, b를 통해 새로운 데이터 세트로 부터 예상 결과 출력
    for data in newDataSet:
        print("Input data: %d, Estimated output: %d" % (data, a * data + b))

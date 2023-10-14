import numpy as np

n = 10000

matrix = np.random.rand(n, n)

while True:
    ans = np.dot(matrix, matrix)

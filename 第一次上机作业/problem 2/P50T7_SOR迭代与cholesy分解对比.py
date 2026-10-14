import math 
import numpy as np 

A = np.array([[4,-1,0], [-1,4,-1], [0,-1,4]])
D = np.diag(np.diag(A))
L = np.array([[0,0,0], [1,0,0], [0,1,0]])
U = np.array([[0,1,0], [0,0,1], [0,0,0]])
b = np.array([1,4,-3])
x_exact = np.linalg.solve(A, b) 
print("Exact solution of the system of equations is: ", x_exact) 
print("\n")

def distance_infinity(x, y):
    return max(abs(x - y))

def SOR(omega):
    x = np.zeros_like(b)
    M = D - omega * L
    N = (1 - omega) * D + omega * U
    for i in range(1,100):
        x = np.linalg.solve(M, N @ x) + omega * np.linalg.solve(M, b)
        if(distance_infinity(x, x_exact) < 5e-6):
            count = i
            break
    return x, count

def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = math.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L

def cholesky_solve(A, b):
    L = cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

def print_vector(name, x):
    """
    统一格式输出向量
    name: 输出标题
    x: numpy数组
   
    """
    print(name)
    print(np.array2string(
        x,
        formatter={'float_kind': lambda value: f"{value:.12f}"}
    ))


x_sor1, count1 = SOR(1.03)
x_sor2, count2 = SOR(1)
x_sor3, count3 = SOR(1.1)

print_vector("SOR with omega = 1.03:", x_sor1)
print("Number of iterations for omega = 1.03:", count1)
print()

print_vector("SOR with omega = 1:", x_sor2)
print("Number of iterations for omega = 1:", count2)
print()

print_vector("SOR with omega = 1.1:", x_sor3)
print("Number of iterations for omega = 1.1:", count3)
print()


x_cholesky = cholesky_solve(A, b)

print_vector("Cholesky solution:", x_cholesky)
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron, bmat, lil_matrix
from scipy.sparse.linalg import spsolve

n = 100
N = n * n
m = 2 * n * (n - 1)  # 网格中的总边数

# 热源位置：数组下标比前面使用的行、列编号小 1
heat = np.zeros((n, n), dtype=bool)
heat[19:23, 67:90] = True
heat[67:87, 18:22] = True
for i in range(40, 68):
    heat[i - 1, i - 1:i + 5] = True

# 固定温度节点：热源，以及最上、最下两行
fixed = heat.copy()
fixed[[0, -1], :] = True

# 构造L = A @ A.T，边界节点只连接网格内的邻点
# remark：以下代码是AI写的，未直接构造A，与书里先构造A再计算L的思路不同，但书写起来更为方便，所以留在这里注释掉了。
#degree = np.full(n, 2.0)
#degree[[0, -1]] = 1
#T = diags([-np.ones(n - 1), degree, -np.ones(n - 1)], [-1, 0, 1])
#L = kron(eye(n), T) + kron(T, eye(n))

A = lil_matrix((N, m))
edge = 0  # 当前边的编号，按从左到右、从上到下的顺序排列，若改变编号规则，并不影响最终L的结果。
for i in range(n):
    for j in range(n):
        k = i * n + j  # 当前节点的编号
        if j < n - 1:  # 当前节点与右侧节点相连
            A[k, edge] = 1
            A[k + 1, edge] = -1
            edge += 1
        if i < n - 1:  # 当前节点与下方节点相连
            A[k, edge] = 1
            A[k + n, edge] = -1
            edge += 1
A = A.tocsr()
L = A @ A.T

# 求解 [I, -L; B, C] @ [s; e] = [0; d]
I = eye(N)
C = diags(fixed.ravel().astype(float))
B = I - C
M = bmat([[I, -L], [B, C]], format='csc')
d = heat.ravel().astype(float)
rhs = np.concatenate([np.zeros(N), d])
s, e = np.split(spsolve(M, rhs), 2)

# 绘图
plt.imshow(e.reshape(n, n), cmap='gray', vmin=0, vmax=1,
           extent=(0.5, 100.5, 100.5, 0.5))
plt.colorbar(label='Temperature')
plt.xlabel('Column')
plt.ylabel('Row')
plt.savefig("heat_diffusion.png", dpi=300, bbox_inches="tight")
plt.show()
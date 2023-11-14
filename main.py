from matrix import matrix, create_band_matrix, solve_jacobi, \
    solve_gauss_seidl, solve_lu_factorization
from math import sin
import time
import matplotlib.pyplot as plt


# Matrix parameters set based on assignment instructions
# 188592
c = 9       # 5-th digit
d = 2       # 6-th digit
e = 5       # 4-th digit
f = 8       # 3-rd digit
N = 992     # 9cd


# Iterative methods, iterations
a1 = 5 + e
a2 = a3 = -1
A = create_band_matrix(a1, a2, a3, N)

b = matrix(N, 1)
for i in range(N):
    b.matrix[i][0] = sin(i * (f + 1))

x0arr = [[1] for _ in range(N)]
x0 = matrix(N, 1, x0arr)

jacobi_x, jacobi_iter, jacobi_norm_res_list = solve_jacobi(A, b, x0, 1e-9)
print("Jacobi iterations:", jacobi_iter)

gauss_seidl_x, gauss_seidl_iter, gauss_norm_res_list = solve_gauss_seidl(A, b, x0, 1e-9)
print("Gauss-Seidl iterations:", gauss_seidl_iter)

plt.plot(jacobi_norm_res_list)
plt.title("Jacobi method residual error norm, converging")
plt.ylabel("Norm of residual error")
plt.xlabel("Iteration")
plt.yscale("log")
plt.show()

plt.plot(gauss_norm_res_list)
plt.title("Gauss-Seidl method residual error norm, converging")
plt.ylabel("Norm of residual error")
plt.xlabel("Iteration")
plt.yscale("log")
plt.show()

# Iterative methods, convergence
a1 = 3
a2 = a3 = -1
A = create_band_matrix(a1, a2, a3, N)

b = matrix(N, 1)
for i in range(N):
    b.matrix[i][0] = sin(i * (f + 1))

x0arr = [[1] for _ in range(N)]
x0 = matrix(N, 1, x0arr)

jacobi_x, jacobi_iter, jacobi_norm_res_list = solve_jacobi(A, b, x0, 1e-9)
print("Jacobi iterations:", jacobi_iter)

gauss_seidl_x, gauss_seidl_iter, gauss_norm_res_list = solve_gauss_seidl(A, b, x0, 1e-9)
print("Gauss-Seidl iterations:", gauss_seidl_iter)

plt.plot(jacobi_norm_res_list)
plt.title("Jacobi method residual error norm, not converging")
plt.ylabel("Norm of residual error")
plt.xlabel("Iteration")
plt.yscale("log")
plt.show()

plt.plot(gauss_norm_res_list)
plt.title("Gauss-Seidl method residual error norm, not converging")
plt.ylabel("Norm of residual error")
plt.xlabel("Iteration")
plt.yscale("log")
plt.show()

# Direct method
a1 = 3
a2 = a3 = -1
A = create_band_matrix(a1, a2, a3, N)

b = matrix(N, 1)
for i in range(N):
    b.matrix[i][0] = sin(i * (f + 1))

lu_x, lu_norm = solve_lu_factorization(A, b)
print("LU norm of residual error: ", lu_norm)

# Methods comparison
a1 = 5 + e
a2 = a3 = -1

N = [100, 200, 500, 1000, 2000, 3000]
jacobi_time = [None] * len(N)
gauss_seidl_time = [None] * len(N)
lu_decomp_time = [None] * len(N)

for i in range(len(N)):
    A = create_band_matrix(a1, a2, a3, N[i])
    b = matrix(N[i], 1)
    for j in range(N[i]):
        b.matrix[j][0] = sin(j * (f + 1))
    x0arr = [[1] for _ in range(N[i])]
    x0 = matrix(N[i], 1, x0arr)

    jacobi_time[i] = time.time()
    jacobi_x, jacobi_iter, jacobi_norm_res_list = solve_jacobi(A, b, x0, 1e-9)
    jacobi_time[i] = time.time() - jacobi_time[i]

    gauss_seidl_time[i] = time.time()
    gauss_seidl_x, gauss_seidl_iter, gauss_norm_res_list = solve_gauss_seidl(A, b, x0, 1e-9)
    gauss_seidl_time[i] = time.time() - gauss_seidl_time[i]

    lu_decomp_time[i] = time.time()
    lu_x, lu_norm = solve_lu_factorization(A, b)
    lu_decomp_time[i] = time.time() - lu_decomp_time[i]

plt.plot(N, jacobi_time)
plt.title("Jacobi method execution time")
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.xticks(N)
plt.show()

plt.plot(N, gauss_seidl_time)
plt.title("Gauss-Seidl method execution time")
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.xticks(N)
plt.show()

plt.plot(N, lu_decomp_time)
plt.title("LU decomposition method execution time")
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.xticks(N)
plt.show()

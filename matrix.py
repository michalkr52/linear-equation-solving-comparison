class matrix:
    def __init__(self, row=0, col=0, mat=None):
        if mat is None:
            self.matrix = [[0 for _ in range(col)] for _ in range(row)]
            self.row = row
            self.col = col
        else:
            self.matrix = mat
            self.row = len(mat)
            self.col = len(mat[0])

    def __str__(self):
        s = ""
        for i in range(self.row):
            s += self.matrix[i].__str__() + "\n"
        return s

    def __add__(self, other):
        result = matrix(self.row, self.col)
        for i in range(self.row):
            for j in range(self.col):
                result.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return result

    def __neg__(self):
        result = matrix(self.row, self.col)
        for i in range(self.row):
            for j in range(self.col):
                result.matrix[i][j] = -self.matrix[i][j]
        return result

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        result = matrix(self.row, other.col)
        for i in range(self.row):
            for j in range(other.col):
                for k in range(self.col):
                    result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return result


def create_band_matrix(a1, a2, a3, N):
    A = matrix(N, N)
    for i in range(N):
        A.matrix[i][i] = a1
        if i - 1 >= 0:
            A.matrix[i][i - 1] = a2
        if i - 2 >= 0:
            A.matrix[i][i - 2] = a3
        if i + 1 < N:
            A.matrix[i][i + 1] = a2
        if i + 2 < N:
            A.matrix[i][i + 2] = a3
    return A


def split_matrix(A):
    L = matrix(A.row, A.col)
    U = matrix(A.row, A.col)
    for i in range(A.row):
        for j in range(A.col):
            if i > j:
                L.matrix[i][j] = A.matrix[i][j]
            elif i == j:
                L.matrix[i][j] = A.matrix[i][j]
                U.matrix[i][j] = A.matrix[i][j]
            else:
                U.matrix[i][j] = A.matrix[i][j]
    return L, U


def lu_factorization(A):
    L = matrix(A.row, A.col)
    U = matrix(A.row, A.col)
    for i in range(A.row):
        for j in range(A.col):
            if i == j:
                L.matrix[i][j] = 1
            U.matrix[i][j] = A.matrix[i][j]
    for k in range(A.row - 1):
        for j in range(k + 1, A.row):
            L.matrix[j][k] = U.matrix[j][k] / U.matrix[k][k]
            for i in range(k, A.row):
                U.matrix[j][i] -= L.matrix[j][k] * U.matrix[k][i]
    return L, U


def forward_substitution(L, b):
    x = matrix(b.row, 1)
    for i in range(b.row):
        x.matrix[i][0] = b.matrix[i][0]
        for j in range(i):
            x.matrix[i][0] -= L.matrix[i][j] * x.matrix[j][0]
        x.matrix[i][0] /= L.matrix[i][i]
    return x


def backward_substitution(U, b):
    x = matrix(b.row, 1)
    for i in range(b.row - 1, -1, -1):
        x.matrix[i][0] = b.matrix[i][0]
        for j in range(i + 1, b.row):
            x.matrix[i][0] -= U.matrix[i][j] * x.matrix[j][0]
        x.matrix[i][0] /= U.matrix[i][i]
    return x


def residual(A, x, b):
    res = matrix(b.row, 1)
    for i in range(b.row):
        res.matrix[i][0] = -b.matrix[i][0]
        for j in range(b.row):
            res.matrix[i][0] += A.matrix[i][j] * x.matrix[j][0]
    return res


def norm(x):
    n = 0
    for i in range(x.row):
        n += x.matrix[i][0] ** 2
    return n ** 0.5


def solve_jacobi(A, b, x0, epsilon):
    x = matrix(x0.row, 1)
    iterations = 0
    upper_bound = 1e10
    norm_res_list = []
    for i in range(x.row):
        x.matrix[i][0] = x0.matrix[i][0]
    while True:
        x1 = matrix(x.row, 1)
        for i in range(x.row):
            x1.matrix[i][0] = b.matrix[i][0]
            for j in range(x.row):
                if j != i:
                    x1.matrix[i][0] -= A.matrix[i][j] * x.matrix[j][0]
            x1.matrix[i][0] /= A.matrix[i][i]
        iterations += 1
        norm_res = norm(residual(A, x1, b))
        norm_res_list.append(norm_res)
        if norm_res < epsilon:
            break
        if norm_res > upper_bound:
            print("Jacobi method does not converge, stopping")
            break
        for i in range(x.row):
            x.matrix[i][0] = x1.matrix[i][0]
    return x1, iterations, norm_res_list


def solve_gauss_seidl(A, b, x0, epsilon):
    x = matrix(x0.row, 1)
    iterations = 0
    upper_bound = 1e10
    norm_res_list = []
    for i in range(x.row):
        x.matrix[i][0] = x0.matrix[i][0]
    while True:
        x1 = matrix(x.row, 1)
        for i in range(x.row):
            x1.matrix[i][0] = b.matrix[i][0]
            for j in range(x.row):
                if j != i:
                    if j < i:
                        x1.matrix[i][0] -= A.matrix[i][j] * x1.matrix[j][0]
                    else:
                        x1.matrix[i][0] -= A.matrix[i][j] * x.matrix[j][0]
            x1.matrix[i][0] /= A.matrix[i][i]
        iterations += 1
        norm_res = norm(residual(A, x1, b))
        norm_res_list.append(norm_res)
        if norm_res < epsilon:
            break
        if norm_res > upper_bound:
            print("Gauss-Seidl method does not converge, stopping")
            break
        for i in range(x.row):
            x.matrix[i][0] = x1.matrix[i][0]
    return x1, iterations, norm_res_list


def solve_lu_factorization(A, b):
    L, U = lu_factorization(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x, norm(residual(A, x, b))

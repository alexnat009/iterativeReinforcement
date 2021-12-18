import numpy as np

A = np.array([[111.5, 2.5, 3.5], [-1.5, -555.5, -1.5], [-14.5, -1.5, 777.5]]).astype(float)
rightB = np.array([-15.1, 15.2, 6.3]).astype(float)


def gauss_seidel(matrix, b, tolerance=1e-12, max_iterations=1000):
    x = np.zeros_like(b, dtype=np.double)
    for k in range(max_iterations):
        x_old = x.copy()
        for i in range(matrix.shape[0]):
            x[i] = (b[i] - np.dot(matrix[i, :i], x[:i]) - np.dot(matrix[i, (i + 1):], x_old[(i + 1):])) / matrix[i, i]

        if np.linalg.norm(x - x_old, ord=1) / np.linalg.norm(x, ord=1) < tolerance:
            break

    return x


def iteration_computation(matrix, f, b, tolerance=1e-10, max_iteration=1000):
    n = matrix.shape[0]
    x = np.ones(n)
    for k in range(max_iteration):
        x_old = x.copy()
        for i in range(n):
            x = f + np.dot(b, x_old).reshape(n, 1)
        if np.linalg.norm(x - x_old, 2) / np.linalg.norm(x, 2) < tolerance:
            break
    return x


def gauss_seidel_matrix_form(matrix, b):
    tmp_L = np.tril(matrix)
    tmp_L_inv = np.linalg.inv(tmp_L)
    tmp_B_gs = np.eye(matrix.shape[0]) - np.dot(tmp_L_inv, matrix)
    tmp_f_gs = np.dot(tmp_L_inv, b)
    return iteration_computation(matrix, tmp_f_gs, tmp_B_gs)


def jacobi_matrix_form(matrix, b):
    tmp_D = np.zeros(matrix.shape)
    np.fill_diagonal(tmp_D, np.diag(matrix))
    tmp_D_inv = np.linalg.inv(tmp_D)
    tmp_B_j = np.eye(matrix.shape[0]) - np.dot(tmp_D_inv, matrix)
    tmp_f_j = np.dot(tmp_D_inv, b)
    return iteration_computation(matrix, tmp_f_j, tmp_B_j)


'''For testing purposes I am comparing gauss_seidel function to in-built np.linealg.solve function '''
print(np.linalg.solve(A, rightB))
print(gauss_seidel(A, rightB))

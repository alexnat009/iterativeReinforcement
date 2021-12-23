import numpy as np


def gauss_seidel(A_matrix, b, tolerance=1e-12, max_iterations=1000):
    x = np.zeros_like(b, dtype=np.double)
    for k in range(max_iterations):
        x_old = x.copy()
        for i in range(A_matrix.shape[0]):
            x[i] = (b[i] - np.dot(A_matrix[i, :i], x[:i]) - np.dot(A_matrix[i, (i + 1):], x_old[(i + 1):])) / A_matrix[i, i]

        if np.linalg.norm(x - x_old, ord=1) / np.linalg.norm(x, ord=1) < tolerance:
            break

    return x


def iteration_computation(A_matrix, f, b, tolerance=1e-10, max_iteration=1000):
    n = A_matrix.shape[0]
    x = np.ones(n)
    for k in range(max_iteration):
        x_old = x.copy()
        for i in range(n):
            x = f + np.dot(b, x_old)
        if np.linalg.norm(x - x_old, 2) / np.linalg.norm(x, 2) < tolerance:
            break
    return x


def gauss_seidel_matrix_form(A_matrix, b):
    tmp_L = np.tril(A_matrix)
    tmp_L_inv = np.linalg.inv(tmp_L)
    tmp_B_gs = np.eye(A_matrix.shape[0]) - np.dot(tmp_L_inv, A_matrix)
    tmp_f_gs = np.dot(tmp_L_inv, b)
    return iteration_computation(A_matrix, tmp_f_gs, tmp_B_gs)


def jacobi_matrix_form(A_matrix, b):
    tmp_D = np.zeros(A_matrix.shape)
    np.fill_diagonal(tmp_D, np.diag(A_matrix))
    tmp_D_inv = np.linalg.inv(tmp_D)
    tmp_B_j = np.eye(A_matrix.shape[0]) - np.dot(tmp_D_inv, A_matrix)
    tmp_f_j = np.dot(tmp_D_inv, b)
    return iteration_computation(A_matrix, tmp_f_j, tmp_B_j)


def richardson_method_matrix_form(A_matrix, P_matrix, omega, b):
    P_matrix_inv = np.linalg.inv(P_matrix)
    tmp_B_rich = np.eye(A_matrix.shape[0]) - omega * np.dot(P_matrix_inv, A_matrix)
    tmp_f_rich = np.dot(P_matrix_inv, omega * b)
    return iteration_computation(A_matrix, tmp_f_rich, tmp_B_rich)

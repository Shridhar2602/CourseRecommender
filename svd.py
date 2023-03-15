import numpy as np

class SVD:
    def __init__(self, A):
        self.u, self.s, self.vh = self.svd(A)

    def gram_schmidt(self, A):
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))

        for j in range(n):
            v = A[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v = v - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            Q[:, j] = v / R[j, j]

        return Q, R

    def qr_eigen(self, A, max_iter=1000, tol=1e-6):

        for _ in range(max_iter):
            Q, R = self.gram_schmidt(A)
            A = np.dot(R, Q)

            if np.amax(np.abs(A)) < tol:
                break

        return np.diag(A), Q

    def svd(self, A):
        m, n = A.shape

        ATA = A.T.dot(A)
        eigenvals, V = self.qr_eigen(ATA)

        idx = eigenvals.argsort()[::-1]
        V = V[:, idx]

        singularvals = np.sqrt(eigenvals[idx])
        S = np.zeros((m, n))
        S[:min(m, n), :min(m, n)] = np.diag(singularvals)

        U = np.zeros((m, m))
        for i in range(min(m, n)):
            if singularvals[i] != 0:
                U[:, i] = A.dot(V[:, i]) / singularvals[i]

        if m >= n:
            return U, S, V.T
        else:
            return V, S.T, U.T

    def return_matrices(self):
        return self.u, self.s, self.vh




import numpy as np
from numpy.linalg import eig

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
			R[j, j] = np.linalg.norm(v, ord=2)
			Q[:, j] = v / R[j, j]

		return Q, R

	def eig(self, A, max_iter=1000, tol=1e-6):
		for _ in range(max_iter):
			Q, R = self.gram_schmidt(A)
			A = np.dot(R, Q)
			
			if np.linalg.norm(A - np.diag(np.diag(A)), ord=2) < tol:
				break

		return np.diag(A), Q

	def svd(self, A):
		m, n = A.shape

		if(m < n):
			A = A.T
			m, n = A.shape

		ATA = A.T.dot(A)
		eigenvals, V = eig(ATA)


		idx = eigenvals.argsort()[::-1]
		self.num_eig = idx.shape[0]
		V = V[:, idx]

		singularvals = np.sqrt(np.maximum(eigenvals[idx], 0))
		self.singularvalues = singularvals
		S = np.zeros((m, n))

		S[:min(m, n), :min(m, n)] = np.diag(singularvals)

		U = np.zeros((m, m))
		for i in range(min(m, n)):
			if singularvals[i] != 0:
				U[:, i] = A.dot(V[:, i]) / singularvals[i]

		return V, S.T, U.T

	def truncated_svd(self, A, k):
		U, S, Vt = self.svd(A)
		
		k = self.num_eig - k
		Uk = U[:, :k]
		Sk = S[:k, :k]
		Vtk = Vt[:k, :]

		return Uk, Sk, Vtk

	def return_matrices(self):
		return self.u, self.s, self.vh

	def return_trunc_matrices(self, A, k):
		return self.truncated_svd(A, k)
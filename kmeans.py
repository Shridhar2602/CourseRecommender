import numpy as np

class K_Means():

	def __init__(self, n_clusters, n_init) -> None:
		self.n_clusters = n_clusters
		self.n_init = n_init

	def fit(self, X, n_iter):
		self.clusters = self._get_clusters(X)
		self.c_assignment = []
		self.inertia = np.Inf

		# print("Cluster Size \t\t\t\t Inertia")
		for _ in range(self.n_init):
			clusters = self._get_clusters(X)
			c_assignment = self._assign_cluster(clusters, X)

			for _ in range(n_iter):
				clusters = self._new_clusters(c_assignment, X)
				c_assignment = self._assign_cluster(clusters, X)

			new_inertia = self.calc_inertia(c_assignment, clusters, X)
			# print(np.unique(c_assignment, return_counts=True)[1], "\t\t\t\t", new_inertia)

			if(new_inertia < self.inertia):
				self.clusters = clusters
				self.c_assignment = c_assignment
				self.inertia = new_inertia

		# print("\nFinal Result -\n", np.unique(self.c_assignment, return_counts=True)[1], "\t\t\t\t", self.calc_inertia(self.c_assignment, self.clusters, X))

	def _assign_cluster(self, clusters, X):

		cluster_assignment = np.ones(X.shape[0], dtype=np.int8)
		for ind, i in enumerate(X):
			cluster_assignment[ind] = self.find_cluster(clusters, i)

		return cluster_assignment

	def find_cluster(self, clusters, x):
		
		dist = []
		for j in clusters:
			dist.append(np.linalg.norm(x - j))

		return dist.index(min(dist))

	def _new_clusters(self, cluster_assignment, X):

		clusters = []
		for i in range(self.n_clusters):
			temp = X[np.where(cluster_assignment == i)]
			
			if(temp.shape[0] == 0):
				temp = X[np.random.randint(0, X.shape[0], 1)[0]].reshape([1, -1])
		
			clusters.append(np.mean(temp, axis=0))

		return clusters

	def get_cluster_size(self):
		return np.unique(self.c_assignment, return_counts=True)[1]


	def calc_inertia(self, c_assignment, clusters, X):
		inertia = 0
		for ind, i in enumerate(X):
			inertia += np.linalg.norm(clusters[c_assignment[ind]] - i)**2

		return inertia

	def _get_clusters(self, X):
		return X[np.random.randint(0, X.shape[0], self.n_clusters), :]
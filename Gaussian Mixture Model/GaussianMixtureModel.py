import numpy as np
from sklearn.cluster import KMeans


class GaussianDistribution:

    def __init__(self, n_clusters, n_epochs):
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
    
    def gaussian(self, X, mu, cov):
        ''' here we implement the Gaussian Density function '''
        n = X.shape[1]
        diff = (X - mu).T
        return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1) 
    

    #Step 1: (Intialization)
    def initialize_clusters(self, X):

        ''' This is the initialization step of the GMM. At this point, we must initialise our parameters  μk, πk and Σk. Here we'll be using results of KMeans as an initial value for μk , πk to one over the number of clusters and Σk to identity matrix. 
        NOTE: We could also use random numbers for everything, but using a sensible initialisation procedure will help the algorithm achieve better results.
         '''
        
        clusters = []
        idx = np.arange(X.shape[0])
        
        # We use the KMeans centroids to initialise the GMM
        
        kmeans = KMeans(self.n_clusters).fit(X)
        mu_k = kmeans.cluster_centers_
        
        for i in range(self.n_clusters):
            clusters.append({
                'pi_k': 1.0 / self.n_clusters,
                'mu_k': mu_k[i],
                'cov_k': np.identity(X.shape[1], dtype=np.float64)
            })
            
        return clusters

    #Step 2 (Expectation step)
    def expectation_step(self, X, clusters):

        ''' Here we calculate the value of ⲅ.
            For simplicity, we just calculate the denominator as a sum over all terms in the numerator, and then assign it to a variable named totals
         '''

        totals = np.zeros((X.shape[0], 1), dtype=np.float64)
        
        for cluster in clusters:
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']
            
            gamma_nk = (pi_k * self.gaussian(X, mu_k, cov_k)).astype(np.float64)
            
            for i in range(X.shape[0]):
                totals[i] += gamma_nk[i]
            
            cluster['gamma_nk'] = gamma_nk
            cluster['totals'] = totals
            
        
        for cluster in clusters:
            cluster['gamma_nk'] /= cluster['totals']

        
    #Step 3 (Maximization step)
    def maximization_step(self, X, clusters):

        ''' Here the value of parameters μk, πk and Σk are updated '''

        N = float(X.shape[0])
    
        for cluster in clusters:
            gamma_nk = cluster['gamma_nk']
            cov_k = np.zeros((X.shape[1], X.shape[1]))
            
            N_k = np.sum(gamma_nk, axis=0)
            
            pi_k = N_k / N
            mu_k = np.sum(gamma_nk * X, axis=0) / N_k
            
            for j in range(X.shape[0]):
                diff = (X[j] - mu_k).reshape(-1, 1)
                cov_k += gamma_nk[j] * np.dot(diff, diff.T)
                
            cov_k /= N_k
            
            cluster['pi_k'] = pi_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k


    #Let us now determine the log-likelihood of the model.
    def get_likelihood(self, X, clusters):
        sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
        return np.sum(sample_likelihoods)

    
    #Putting everything together
    # 1. Initialise the parameters by using the initialise_clusters function
    # 2. perform several expectation-maximization steps
    def train_gmm(self, X):
        clusters = self.initialize_clusters(X)
        likelihoods = np.zeros((self.n_epochs, ))
        scores = np.zeros((X.shape[0], self.n_clusters))

        for i in range(self.n_epochs):
            
            self.expectation_step(X, clusters)
            self.maximization_step(X, clusters)

            likelihood = self.get_likelihood(X, clusters)
            likelihoods[i] = likelihood

            print('Epoch: ', i + 1, 'Likelihood: ', likelihood)
            
        for i, cluster in enumerate(clusters):
            scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)
            
        return likelihoods


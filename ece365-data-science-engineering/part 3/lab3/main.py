import numpy as np


class Question1(object):
    def pca_reduce_dimen(self, X):
        
        ### size of data is L x n, where L is the dimension and n is the # of samples ###
        L = X.shape[0]
        n = X.shape[1]
        k = 2 # number of principal axes

        ### Sample mean ###
        mean = np.mean(X, axis=1) # vector of length L

        ### Covariance matrix ###
#         cov_matrix = np.cov(X) # size of n x n

        ### Covariance matrix formula ###
        X_minus_mean = np.zeros_like(X)

        for i in range(L):
            X_minus_mean[i] = X[i] - mean[i]

        cov_matrix = (X_minus_mean @ X_minus_mean.T)/n

        ### Eigenvalue Decomposition ###
        eigval, eigvec = np.linalg.eig(cov_matrix)
        
        ### Principal Axes and Components ###
        indices = np.argsort(eigval) # last element is the largest, sorted in ascending order
        components = np.zeros((2,n))
        
        for i in range(k): # iterate over number of principal axes
            prin_axis = eigvec[:,indices[-(i+1)]] # grab eigenvectors corresponding to largest eigenvalues
            
            for j in range(n):
                components[i,j] = np.inner(X[:,j], prin_axis)
            
        return components

    def pca_project(self, X, k):
        
        ### size of data is L x n, where L is the dimension and n is the # of samples ###
        L = X.shape[0]
        n = X.shape[1]
#         k = 2 # number of principal axes

        ### Sample mean ###
        mean = np.mean(X, axis=1) # vector of length L

        ### Covariance matrix ###
#         cov_matrix = np.cov(X) # size of n x n

        ### Covariance matrix formula ###
        X_minus_mean = np.zeros_like(X)

        for i in range(L):
            X_minus_mean[i] = X[i] - mean[i]

        cov_matrix = (X_minus_mean @ X_minus_mean.T)/n

        ### Eigenvalue Decomposition ###
        eigval, eigvec = np.linalg.eig(cov_matrix)
        
        ### Principal Axes and Components ###
        indices = np.argsort(eigval) # last element is the largest, sorted in ascending order
        components = np.zeros((2,n))
        filtered = np.zeros((L,n))
        
        for i in range(k): # iterate over number of principal axes
            prin_axis = eigvec[:,indices[-(i+1)]].reshape((-1,1)) # grab eigenvectors corresponding to largest eigenvalues
            
            components = np.inner(X.T, prin_axis.T)
            argument   = components @ prin_axis.T
            filtered  += argument.T                
            
        return filtered


class Question2(object):
    def wiener_filter(self, data_noisy, C, mu, sigma):

        sigma2_I = (sigma**2) * np.identity(C.shape[0])
        inv_term = np.linalg.inv(C + sigma2_I)
        y_min_mu = data_noisy - mu
        
        filtered = mu + (C @ inv_term @ y_min_mu)
        
        return filtered


class Question3(object):
    def embedding(self, A):
        
        eig_val, eig_vec = np.linalg.eigh(A)
        
        eig_val = sorted(eig_val, reverse=True)

        return eig_vec, eig_val

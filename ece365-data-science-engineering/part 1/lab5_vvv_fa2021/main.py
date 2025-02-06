import numpy as np

class Question1(object):
    def pca(self,data):
        """ Implement PCA via the eigendecomposition or the SVD.

        Parameters:
        1. data     (N,d) numpy ndarray. Each row as a feature vector.

        Outputs:
        1. W        (d,d) numpy array. PCA transformation matrix (Note that each **row** of the matrix should be a principal component)
        2. s        (d,) numpy array. Vector consisting of the amount  of variance explained in the data by each PCA feature.
        Note that the PCA features are ordered in **decreasing** amount of variance explained, by convention.
        """
        W = np.zeros((data.shape[1],data.shape[1]))
        s = np.zeros(data.shape[1])
        # Put your code below
        
        # Formula in typed notes on page 82
        cov_matrix = (1 / len(data)) * (data.T @ data) # estimate covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # calculate eigendecomposition
        eigenvectors = eigenvectors.T
        
        # determine PCA transformation
        W = eigenvectors[::-1] # flip matrices
        s = eigenvalues[::-1]

        return (W,s)

    def pcadimreduce(self,data,W,k):
        """ Implements dimension reduction via PCA.

        Parameters:
        1. data     (N,d) numpy ndarray. Each row as a feature vector.
        2. W        (d,d) numpy array. PCA transformation matrix
        3. k        number. Number of PCA features to retain

        Outputs:
        1. reduced_data  (N,k) numpy ndarray, where each row contains PCA features corresponding to its input feature.
        """
        reduced_data = np.zeros((data.shape[0],k))
        # Put your code below
        
        # Formula in typed notes on page 82
        Wk = W[:k] # first k rows of W                                      
        reduced_data = data @ Wk.T
    
        return reduced_data

    def pcareconstruct(self,pcadata,W):
        """ Implements dimension reduction via PCA.

        Parameters:
        1. pcadata  (N,k) numpy ndarray. Each row as a PCA vector. (e.g. generated from pcadimreduce)
        2. W        (d,d) numpy array. PCA transformation matrix

        Outputs:
        1. reconstructed_data  (N,d) numpy ndarray, where the i-th row contains the reconstruction of the original i-th input feature vector (in `data`) based on the PCA features contained in `pcadata`.
        """
        reconstructed_data = np.zeros((pcadata.shape[0],W.shape[0]))
        # Put your code below
        
        # Formula in typed notes on page 82
        Wk = W[:pcadata.shape[1]] # first k rows of W   
        reconstructed_data = pcadata @ Wk
        
        return reconstructed_data

from sklearn.decomposition import PCA

class Question2(object):
    def unexp_var(self,X,k):
        """Returns an numpy array with the fraction of unexplained variance on X by retaining the first k principal components for k =1,...200.
        Parameters:
        1. X        The input image

        Returns:
        1. pca      The PCA object fit on X
        2. unexpv   A (k,) numpy vector, where the i-th element contains the percentage of unexplained variance on X by retaining i+1 principal components
        """
        pca = None
        unexpv = np.zeros(k)
        # Put your code below
  
        # Load and fit PCA model
        pca = PCA(n_components=k)
        pca.fit(X)
        
        # Find unexplained variance
        exp_var_ratio = pca.explained_variance_ratio_
        expv = exp_var_ratio.cumsum() # cummulative explained variance
        unexpv = 1 - expv
        
        return (pca,unexpv)

    def pca_approx(self,X_t,pca,i):
        """Returns an approimation of `X_t` using the the first `i`  principal components (learned from `X`).

        Parameters:
            1. X_t      The input image to be approximated
            2. pca      The PCA object to use for the transform
            3. i        Number of principal components to retain

        Returns:
            1. recon_img    The reconstructed approximation of X_t using the first i principal components learned from X (As a sanity check it should be of size (1,4096))
        """
        recon_img = np.zeros((1,4096))
        # Put your code below

        transformed_Xt = pca.transform(X_t.reshape(1,-1)) # transform input image
        
        retain_comp = np.zeros(transformed_Xt.shape)
        for n in range(i):
            retain_comp[:,n] = transformed_Xt[:,n]# transformed image w/ retain principal components i
            
        recon_img = pca.inverse_transform(retain_comp.reshape(1,-1)) # reconstructed image

        return recon_img

from sklearn import neighbors

class Question3(object):
    def pca_classify(self,traindata,trainlabels,valdata,vallabels,k):
        """Returns validation errors using 1-NN on the PCA features using 1,2,...,k PCA features, the minimum validation error, and number of PCA features used.

        Parameters:
            1. traindata       (Nt, d) numpy ndarray. The features in the training set.
            2. trainlabels     (Nt,) numpy array. The responses in the training set.
            3. valdata         (Nv, d) numpy ndarray. The features in the validation set.
            4. valabels        (Nv,) numpy array. The responses in the validation set.
            5. k               Integer. Maximum number of PCA features to retain

        Returns:
            1. ve              A length k numpy array, where ve[i] is the validation error using the first i+1 features (i=0,...,255).
            2. min_ve          Minimum validation error
            3. min_pca_feat    Number of PCA features to retain. Integer.
        """

        ve = np.zeros(k)
        # Put your code below

        # code provided from lab 3
        error = lambda y, yhat: np.mean(y!=yhat)

        # code written from lab 2
        nn = neighbors.KNeighborsClassifier(n_neighbors=1) # adjust model parameters
        
        k_vals = np.arange(1,k+1) # all possible k values       
        min_ve = 99999 # inital values for finding min val err and min PCA features
        min_pca_feat = 0
        
        for i, K in enumerate(k_vals):
            pca = PCA(n_components=K) # load and train PCA model with K components
            pca.fit(traindata)
            
            transformed_train_data = pca.transform(traindata) # transform data for 1-NN
            transformed_val_data = pca.transform(valdata)
            
            nn.fit(transformed_train_data, trainlabels)  # train 1-NN model
            val_pred = nn.predict(transformed_val_data) # make predctions based on val data
            
            val_err = error(vallabels, val_pred) # calculate and store val errors
            ve[i] = val_err
            
            if val_err < min_ve:
                min_ve = val_err
                min_pca_feat = K
                
            if K%50 == 0:
                print(K)

        return (ve, min_ve, min_pca_feat)

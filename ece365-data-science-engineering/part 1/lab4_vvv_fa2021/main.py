import numpy as np
from sklearn import neighbors
import scipy.spatial.distance as dist
from sklearn import linear_model
from sklearn.model_selection import train_test_split

class Question1(object):
    def kMeans(self,data,K,niter):
        """ Implement the K-Means algorithm.

        **For grading purposes only:**

        Do NOT change the random seed, otherwise we are not able to grade your code! This is true throughout this script. However, in practice, you never want to set a random seed like this.
        For your own interest, after you have finished implementing this function, you can change the seed to different values and check your results.
        Please use numpy library for initial random choice. This will use the seed above. Scipy library is using a different seeding system, so that would probably result in an error during our grading.

        Parameters:
        1. data     (N, d) numpy ndarray. The unlabelled data with each row as a feature vector.
        2. K        Integer. It indicates the number of clusters.
        3. niter    Integer. It gives the number of iterations. An iteration includes an assignment process and an update process.

        Outputs:
        1. labels   (N,) numpy array. It contains which cluster (0,...,K-1) a feature vector is in. It should be the (niter+1)-th assignment.
        2. centers  (K, d) numpy ndarray. The i-th row should contain the i-th center.
        """
        np.random.seed(12312)
        # Put your code below
          
        # Initialize centers of the K clusters
        rand_idx = len(data)
        rand_idx = np.random.choice(rand_idx, K, replace=False)
        centers = data[rand_idx]     
        
        # K-means algorithm in typed notes on page 44
        labels = np.zeros(len(data))
        for i in range(niter):    
            for j in range(len(data)): # assign xi to cluster zi
                distance = dist.cdist(centers, [data[j]], metric='euclidean')
                labels[j] = np.argmin(distance)

            for k in range(K): # update center of cluster k
                k_labels = data[labels == k]
                centers[k] = k_labels.mean(axis=0)
            
        # Remember to check your data types: labels should be integers!
        return (labels, centers)

    def calculateJ(self,data,kMeansfun):
        """ Calculate the J_k value for K=2,...,10.

        This function should call the given kMeansfun function and set niter=100.

        Parameters:
        1. data         (N, d) numpy ndarray. The unlabelled data with each row as a feature vector.
        2. kMeansfun    Correct kMeans function. You should use kMeansfun wherever you want to use kMeans in function calculateJ. This is used for grading purposes.

        Outputs:
        1. err          (9,) numpy array. The i-th element contains the J_k value when k = i+2.
        """
        err = np.zeros(9)
        # Put your code below      
        niter = 100
        K = range(2,11)
        
        # Picking K algorithm in typed notes on page 45
        for i, k in enumerate(K):
            labels, centers = kMeansfun(data, k, niter)
            labels = labels.astype(int)
            
            for j in range(len(data)):
                z = labels[j]
                x = data[j]                
                u = centers[z]
                err[i] += sum((x - u)**2)
                
        return err

from sklearn.cluster import KMeans

class Question2(object):
    def trainVQ(self,image,B,K):
        """ Generate a codebook for vector quantization.

        Please use the KMeans function from the sklearn package. You can use kmeans.cluster_centers_ to get the cluster centers after you fit your model.

        For grading purposes only: Please flatten any matrix in *row-major* order. If you prefer, you can use np.flatten(xxx) to flatten your matrix.

        Parameters:
        1. image        (N, M) numpy ndarray. It represents a grayscale image.
        2. B            Integer. You will use B×B blocks for vector quantization. You may assume that both N and M are divisible by B.
        3. K            Integer. It gives the size of your codebook.

        Outputs:
        1. codebook     (K, B^2) numpy ndarray. It is the codebook you should return.
        2. kmeans       KMeans Object. For grading only.
        """
        np.random.seed(12345)
        # Put your code below
        
        # Training a Vector Quantizer in typed notes on page 50
        
        # Initialize blocks
        N = image.shape[0]
        M = image.shape[1]      
        size_of_block = B**2 
        total_blocks = (N*M)//size_of_block
                
        # Partition image into blocks
        block_rows = N//B
        block_cols = M//B
        data_blocks = np.zeros((total_blocks, size_of_block))       

        # Store image into data blocks
        for rows in range(block_rows):
            start_row = rows * B # indexes for rows
            end_row = (rows + 1) * B
            
            for cols in range(block_cols):
                start_col = cols * B # indexes for cols
                end_col = (cols + 1) * B
                
                data_block = image[start_row:end_row, start_col:end_col] # one block of data
                data_blocks[(rows * block_cols) + cols] = data_block.flatten() # row major order index
        
                        
        # Load and train model
        kmeans = KMeans(n_clusters=K, init='k-means++')
        kmeans.fit(data_blocks)       
        codebook = kmeans.cluster_centers_
        
        return (codebook,kmeans)

    def compressImg(self,image,codebook,B):
        """ Compress an image using a given codebook.

        You can use the nearest neighbor classifier from scikit-learn if you want (though it is not necessary) to map blocks to their nearest codeword.

        **For grading purposes only:**

        Please flatten any matrix in *row-major* order. If you prefer, you can use np.flatten(xxx) to flatten your matrix.

        Parameters:
        1. image        (N, M) numpy ndarray. It represents a grayscale image. You may assume that both N and M are divisible by B.
        2. codebook     (K, B^2) numpy ndarray. The codebook used in compression.
        3. B            Integer. Block size.

        Outputs:
        1. cmpimg       (N//B, M//B) numpy ndarray. It consists of the indices in the codebook used to approximate the image.
        """
        # Put your code below        
        
        # Initialize blocks
        N = image.shape[0]
        M = image.shape[1]      
        size_of_block = B**2 
        total_blocks = (N*M)//size_of_block
                
        # Partition image into blocks
        block_rows = N//B
        block_cols = M//B
        data_blocks = np.zeros((total_blocks, size_of_block))       

        # Store image into data blocks
        for rows in range(block_rows):
            start_row = rows * B # indexes for rows
            end_row = (rows + 1) * B
            
            for cols in range(block_cols):
                start_col = cols * B # indexes for cols
                end_col = (cols + 1) * B
                row_major_idx = (rows * block_cols) + cols  # row major order index
                
                data_block = image[start_row:end_row, start_col:end_col] # one block of data
                data_blocks[row_major_idx] = data_block.flatten() # store data block and flatten array              
        # above code from trainVQ, new code below      
                    
                
        # Map data blocks to their nearest codeword
        labels = np.arange(len(codebook))
        nn = neighbors.KNeighborsClassifier(n_neighbors=1) # Load and train model
        nn.fit(codebook, labels)
        
        cmpimg = np.zeros((block_rows, block_cols))
        for rows in range(block_rows):  
            for cols in range(block_cols):
                row_major_idx = (rows * block_cols) + cols  # row major order index
                
                image = data_blocks[row_major_idx]
                cmpimg[rows, cols] = nn.predict(image.reshape(1,-1)) # compressed image
        
        # Check that your indices are integers!
        return cmpimg.astype(int)

    def decompressImg(self,indices,codebook,B):
        """ Reconstruct an image from its codebook.

        You can use np.reshape() to reshape the flattened array.

        Parameters:
        1. indices      (N//B, M//B) numpy ndarray. It contains the indices of the codebook for each block.
        2. codebook     (K, B^2) numpy ndarray. The codebook used in compression.
        3. B            Integer. Block size.

        Outputs:
        1. rctimage     (N, M) numpy ndarray. It consists of the indices in the codebook used to approximate the image.
        """
        # Put your code below
        
        # Initialize blocks
        block_rows = indices.shape[0]
        block_cols = indices.shape[1]
        N = block_rows*B
        M = block_cols*B
        
        size_of_block = B**2
        total_blocks = (N*M)//size_of_block   
        data_blocks = np.zeros((total_blocks, size_of_block))
        rctimage = np.zeros((N,M))

        # Reconstruct data blocks from codebook
        for rows in range(block_rows):
            for cols in range(block_cols):
                row_major_idx = (rows * block_cols) + cols  # row major order index
                codebook_idx = indices[rows, cols]
                data_blocks[row_major_idx] = codebook[codebook_idx]

        # Reconstruct image from data blocks
        for rows in range(block_rows):
            start_row = rows * B # indexes for rows
            end_row = (rows + 1) * B           
            
            for cols in range(block_cols):
                start_col = cols * B # indexes for cols
                end_col = (cols + 1) * B
                row_major_idx = (rows * block_cols) + cols  # row major order index
                
                data_block = data_blocks[row_major_idx] # one block of data
                data_block = data_block.reshape(B,B) # unflatten array
                rctimage[start_row:end_row, start_col:end_col] = data_block # reconstructed image

        return rctimage

class Question3(object):
    def generatePrototypes(self,traindata,trainlabels,K_list):
        """ Generate prototypes from labeled data.

        You can use the KMeans function from the sklearn package.

        **For grading purposes only:**

        Do NOT change the random seed, otherwise we are not able to grade your code!

        Parameters:
        1. traindata        (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels      (Nt,) numpy array. The labels in the training set.
        3. K_list           List. A list of integers corresponding to the number of prototypes under each class.

        Outputs:
        1. proto_dat_list   A length len(K_list) list. The K-th element in the list is a (K * num_classes, d) numpy ndarray, representing the prototypes selected if using K prototypes under each class. You should keep the order as in the given K_list.
        2. proto_lab_list   A length len(K_list) list. The K-th element in the list is a (K * num_classes,) numpy array, representing the corresponding labels if using K prototypes under each class. You should keep the order as in the given K_list.
        """
        np.random.seed(56789)   # As stated before, do NOT change this line!
        proto_dat_list = []
        proto_lab_list = []
        # Put your code below
        
        # Algorithm in typed notes on page 52        
        for K in K_list:
            unique_classes = np.unique(trainlabels) # unique classes
            unique_classes = unique_classes.astype(int) # cast as int for y            
            num_classes = len(unique_classes) # number of unique classes
            class_prototypes = K * num_classes # prototypes of class l
            
            proto_data = np.zeros((class_prototypes, traindata.shape[1])) # grab prototype data and labels
            proto_labels = np.zeros(class_prototypes)
        
            for i in unique_classes:
                kmeans = KMeans(n_clusters = K, init='k-means++') # load and train model
                kmeans.fit(traindata[trainlabels==i])

                proto_data[K*i:K*(i+1)] = kmeans.cluster_centers_ # grab centers and labels
                proto_labels[K*i:K*(i+1)] = i
                
            proto_dat_list.append(proto_data)
            proto_lab_list.append(proto_labels)       
        
        # Check that your proto_lab_list only contains integer arrays!
        return (proto_dat_list, proto_lab_list)

    def protoValError(self,proto_dat_list,proto_lab_list,valdata,vallabels):
        """ Generate prototypes from labeled data.

        You may assume there are at least min(K_list) examples under each class. set(trainlabels) will give you the set of labels.

        Parameters:
        1. proto_dat_list   A list of (K * num_classes, d) numpy ndarray. A list of prototypes selected. This should be one of the outputs from your previous function.
        2. proto_lab_list   A list of (K * num_classes,) numpy array. A list of corresponding labels for the selected prototypes. This should be one of the outputs from your previous function.
        3. valdata          (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels        (Nv,) numpy array. The labels in the validation set.

        Outputs:
        1. proto_err        (len(proto_dat_list),) numpy ndarray. The validation error for each K value (in the same order as the given K_list).
        """
        proto_err = np.zeros(len(proto_dat_list))
        # Put your code below
        
        error = lambda y, yhat: np.mean(y!=yhat) # from lab 3
        
        for i in range(len(proto_dat_list)):                    
            nn = neighbors.KNeighborsClassifier(n_neighbors=1) # Load, train, and predict using NN classifier
            nn.fit(proto_dat_list[i], proto_lab_list[i])
            val_pred = nn.predict(valdata)
            proto_err[i] = error(vallabels, val_pred) # calculate error

        return proto_err

class Question4(object):
    def benchmarkRSS(self,trainfeat,trainresp,valfeat,valresp):
        """ Return the benchmark RSS.

        In particular, always predict the response as zero (mean response on the training data).

        Calculate the validation RSS for this model. Please use the formula as defined in the jupyter notebook.

        Parameters:
        1. trainfeat    (Nt, d) numpy ndarray. The features in the training set.
        2. trainresp    (Nt,) numpy array. The responses in the training set.
        3. valfeat      (Nv, d) numpy ndarray. The features in the validation set.
        4. valresp      (Nv,) numpy array. The responses in the validation set.

        Outputs:
        1. rss          Scalar. The validation RSS.
        """
        # Put your code below
         
        # Formula in typed notes on page 57        
        V = len(valfeat) # number of samples       
        B = np.zeros(valfeat.shape[1]) # weights are zero
     
        rss = (1 / V) * sum((valresp - (valfeat @ B)) ** 2) # baseline rss

        return rss

    def OLSRSS(self,trainfeat,trainresp,valfeat,valresp):
        """ Return the RSS from the ordinary least squares model.

        Use sklearn.linear_model.LinearRegression() with the default parameters.

        Calculate the validation RSS for this model. Please use the formula as defined in the jupyter notebook.

        Note: The .score() method returns an  R^2 value, not the RSS, so you shouldn't use it anywhere in this problem.

        Parameters:
        1. trainfeat    (Nt, d) numpy ndarray. The features in the training set.
        2. trainresp    (Nt,) numpy array. The responses in the training set.
        3. valfeat      (Nv, d) numpy ndarray. The features in the validation set.
        4. valresp      (Nv,) numpy array. The responses in the validation set.

        Outputs:
        1. rss          Scalar. The validation RSS.
        """
        # Put your code below
        V = len(valfeat) # number of samples
        
        regression = linear_model.LinearRegression()
        regression.fit(trainfeat, trainresp)
        val_pred = regression.predict(valfeat)
        
        rss = (1 / V) * sum((val_pred - valresp) ** 2)

        return rss

    def RidgeRSS(self,trainfeat,trainresp,valfeat,valresp):
        """ Return the RSS from the ridge regression.

        Apply ridge regression with sklearn.linear_model.Ridge. Sweep the regularization/tuning parameter α = 0,...,100 with 1000 equally spaced values.

        Note: Larger values of α shrink the weights in the model more, and α=0 corresponds to the LS solution.

        Parameters:
        1. trainfeat    (Nt, d) numpy ndarray. The features in the training set.
        2. trainresp    (Nt,) numpy array. The responses in the training set.
        3. valfeat      (Nv, d) numpy ndarray. The features in the validation set.
        4. valresp      (Nv,) numpy array. The responses in the validation set.

        Outputs:
        1. rss_array    (1000,). The validation RSS array. This is used for plotting. This will not be tested by the autograder.
        2. best_a       Scalar. The alpha that minimizes the RSS.
        3. best_rss     Scalar. The corresponding RSS.
        4. coef         (d,) numpy array. The minimizing coefficient. This is for visualization only. This will not be tested by the autograder.
        """
        a = np.linspace(0,100,1000)
        rss_array = np.zeros(a.shape)
        # Put your code below
        
        V = len(valfeat) # number of samples 
        best_rss = 10000 # set a high rss so that it updates later to the lowest rss 
        
        # Find the alpha that gives the least RSS
        for i, alpha in enumerate(a):
            ridge = linear_model.Ridge(alpha=alpha) # Load, train, and predict using ridge model
            ridge.fit(trainfeat, trainresp)
            val_pred = ridge.predict(valfeat)
            
            rss = (1 / V) * sum((val_pred - valresp) ** 2) # Calculate rss for given alpha
            rss_array[i] = rss
            
            # store a and coefficient for best rss 
            if rss < best_rss:
                best_rss = rss
                best_a = alpha
                coef = ridge.coef_
        
        return (rss_array, best_a, best_rss, coef)

    def LassoRSS(self,trainfeat,trainresp,valfeat,valresp):
        """ Return the RSS from the Lasso regression.

        Apply lasso regression with sklearn.linear_model.Lasso. Sweep the regularization/tuning parameter α = 0,...,1 with 1000 equally spaced values.

        Note: Larger values of α will lead to sparser solutions (i.e. less features used in the model), with a sufficiently large value of α leading to a constant prediction. Small values of α are closer to the LS solution, with α=0 being the LS solution.

        Parameters:
        1. trainfeat    (Nt, d) numpy ndarray. The features in the training set.
        2. trainresp    (Nt,) numpy array. The responses in the training set.
        3. valfeat      (Nv, d) numpy ndarray. The features in the validation set.
        4. valresp      (Nv,) numpy array. The responses in the validation set.

        Outputs:
        1. rss_array    (1000,). The validation RSS array. This is used for plotting. This will not be tested by the autograder.
        2. best_a       Scalar. The alpha that minimizes the RSS.
        3. best_rss     Scalar. The corresponding RSS.
        4. coef         (d,) numpy array. The minimizing coefficient. This is for visualization only. This will not be tested by the autograder.
        """
        a = np.linspace(0.00001,1,1000)     # Since 0 will give an error, we use 0.00001 instead.
        rss_array = np.zeros(a.shape)
        # Put your code below
        
        V = len(valfeat) # number of samples 
        best_rss = 10000 # set a high rss so that it updates later to the lowest rss 
        
        # Find the alpha that gives the least RSS
        for i, alpha in enumerate(a):
            lasso = linear_model.Lasso(alpha=alpha) # Load, train, and predict using ridge model
            lasso.fit(trainfeat, trainresp)
            val_pred = lasso.predict(valfeat)
            
            rss = (1 / V) * sum((val_pred - valresp) ** 2) # Calculate rss for given alpha
            rss_array[i] = rss
            
            # store a and coefficient for best rss 
            if rss < best_rss:
                best_rss = rss
                best_a = alpha
                coef = lasso.coef_

        return (rss_array, best_a, best_rss, coef)

import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# You may use this function as you like.
error = lambda y, yhat: np.mean(y!=yhat)

class Question1(object):
    # The sequence in this problem is different from the one you saw in the jupyter notebook. This makes it easier to grade. Apologies for any inconvenience.
    def BernoulliNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a BernoulliNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        
        # load the model and adjust parameters
        classifier = BernoulliNB()        
        
        # fit the model using the training data
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        fittingTime = time.time() - start_time # time taken to fit the model
        
        # make predictions using the training data and validation data
        train_pred = classifier.predict(traindata) # train predictions        
        
        start_time = time.time()
        val_pred = classifier.predict(valdata) # val predictions
        valPredictingTime = time.time() - start_time # time taken to predict the labels
        
        # calculate the training and validation errors
        trainingError = error(trainlabels, train_pred)
        validationError = error(vallabels, val_pred)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def MultinomialNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a MultinomialNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        
        # load the model and adjust parameters
        classifier = MultinomialNB()        
        
        # fit the model using the training data
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        fittingTime = time.time() - start_time # time taken to fit the model
        
        # make predictions using the training data and validation data
        train_pred = classifier.predict(traindata) # train predictions        
        
        start_time = time.time()
        val_pred = classifier.predict(valdata) # val predictions
        valPredictingTime = time.time() - start_time # time taken to predict the labels
        
        # calculate the training and validation errors
        trainingError = error(trainlabels, train_pred)
        validationError = error(vallabels, val_pred)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LinearSVC_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LinearSVC classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        
        # load the model and adjust parameters
        classifier = LinearSVC()        
        
        # fit the model using the training data
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        fittingTime = time.time() - start_time # time taken to fit the model
        
        # make predictions using the training data and validation data
        train_pred = classifier.predict(traindata) # train predictions        
        
        start_time = time.time()
        val_pred = classifier.predict(valdata) # val predictions
        valPredictingTime = time.time() - start_time # time taken to predict the labels
        
        # calculate the training and validation errors
        trainingError = error(trainlabels, train_pred)
        validationError = error(vallabels, val_pred)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LogisticRegression_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LogisticRegression classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        
        # load the model and adjust parameters
        classifier = LogisticRegression()      
        
        # fit the model using the training data
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        fittingTime = time.time() - start_time # time taken to fit the model
        
        # make predictions using the training data and validation data
        train_pred = classifier.predict(traindata) # train predictions        
        
        start_time = time.time()
        val_pred = classifier.predict(valdata) # val predictions
        valPredictingTime = time.time() - start_time # time taken to predict the labels
        
        # calculate the training and validation errors
        trainingError = error(trainlabels, train_pred)
        validationError = error(vallabels, val_pred)        
        
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def NN_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a Nearest Neighbor classifier using the given data.

        Make sure to modify the default parameter.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata              (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels            (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below        
        
        # load the model and adjust parameters
        classifier = KNeighborsClassifier(n_neighbors=1)        
        
        # fit the model using the training data
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        fittingTime = time.time() - start_time # time taken to fit the model
        
        # make predictions using the training data and validation data
        train_pred = classifier.predict(traindata) # train predictions        
        
        start_time = time.time()
        val_pred = classifier.predict(valdata) # val predictions
        valPredictingTime = time.time() - start_time # time taken to predict the labels
        
        # calculate the training and validation errors
        trainingError = error(trainlabels, train_pred)
        validationError = error(vallabels, val_pred)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def confMatrix(self,truelabels,estimatedlabels):
        """ Write a function that calculates the confusion matrix (cf. Fig. 2.1 in the notes).

        You may wish to read Section 2.1.1 in the notes -- it may be helpful, but is not necessary to complete this problem.

        Parameters:
        1. truelabels           (Nv, ) numpy ndarray. The ground truth labels.
        2. estimatedlabels      (Nv, ) numpy ndarray. The estimated labels from the output of some classifier.

        Outputs:
        1. cm                   (2,2) numpy ndarray. The calculated confusion matrix.
        """
        cm = np.zeros((2,2))
        # Put your code below
                        
        true_pos = 0
        false_pos = 0
        false_neg = 0
        true_neg = 0
        
        for i in range(len(truelabels)):
            if truelabels[i] == 1 and estimatedlabels[i] == 1: # check for true positives
                true_pos += 1
            if truelabels[i] == -1 and estimatedlabels[i] == 1: # check for false positives
                false_pos += 1
            if truelabels[i] == 1 and estimatedlabels[i] == -1: # check for false negatives
                false_neg += 1
            if truelabels[i] == -1 and estimatedlabels[i] == -1: # check for true negatives
                true_neg += 1        
        
        cm[0][0] = true_pos
        cm[0][1] = false_pos
        cm[1][0] = false_neg
        cm[1][1] = true_neg

        return cm

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Run the classifier you selected in the previous part of the problem on the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. testError            Float. The reported test error. It should be less than 1.
        3. confusionMatrix      (2,2) numpy ndarray. The resulting confusion matrix. This will not be graded here.
        """
        # Put your code below
        
        # train and return the model + testError
        output = self.LinearSVC_classifier(traindata, trainlabels, testdata, testlabels) # returns a tuple
        classifier = output[0]
        testError = output[2]
        
         # get labels for the confusion matrix   
        estimatedlabels = classifier.predict(testdata)       

        # You can use the following line after you finish the rest
        confusionMatrix = self.confMatrix(testlabels, estimatedlabels)
        # Do not change this sequence!
        return (classifier, testError, confusionMatrix)

class Question2(object):
    def crossValidationkNN(self, traindata, trainlabels, k):
        """ Write a function which implements 5-fold cross-validation to estimate the error of a classifier with cross-validation with the 0,1-loss for k-Nearest Neighbors (kNN).

        For this problem, take your folds to be 0:N/5, N/5:2N/5, ..., 4N/5:N for cross-validation.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. err[i] is the cross-validated estimate of using i neighbors (the zero-th component of the vector will be meaningless).
        """
        err = np.zeros(k+1)
        # Put your code below
        
        num_of_folds = 5
        N = len(traindata)
        index = N//num_of_folds # index needs to be an integer
        
        # choose k for kNN
        for i in range(1,k+1):
            classifier = KNeighborsClassifier(n_neighbors=i) # load model
            
            # train model and make predictions using cross-validation
            for j in range(num_of_folds):
                start_index = j*index
                end_index = (j+1)*index             
                
                val_fold = traindata[start_index:end_index] # grab the validation fold
                val_labels = trainlabels[start_index:end_index]
                                
                bool_array = np.ones(N, dtype=bool) # boolean array to grab the rest of the indices
                bool_array[start_index:end_index] = False
                
                train_folds = traindata[bool_array == True] # grab the training folds
                train_labels = trainlabels[bool_array == True]
                
                classifier.fit(train_folds, train_labels) # fit the model
                val_pred = classifier.predict(val_fold) # make predictions using val fold
                err[i] += error(val_labels, val_pred) / num_of_folds # calculate cross-val error     
        
        return err

    def minimizer_K(self, kNN_errors):
        """ Write a function that calls the above function and returns 1) the output from the previous function, 2) the number of neighbors within  1,...,k  that minimizes the cross-validation error, and 3) the correponding minimum error.

        Parameters:
        1. kNN_errors           (k+1,) numpy ndarray. The output from self.crossValidationkNN()

        Outputs:
        1. k_min                Integer (np.int64 or int). The number of neighbors within  1,...,k  that minimizes the cross-validation error.
        2. err_min              Float. The correponding minimum error.
        """
        # Put your code below
        
        err_min = kNN_errors[1]
        k_min = 1
        for i in range(1,len(kNN_errors)): # find the k-value that has the smallest error
            if err_min > kNN_errors[i]:
                err_min = kNN_errors[i]
                k_min = i
                
        # Do not change this sequence!
        return (k_min, err_min)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train a kNN model on the whole training data using the number of neighbors you found in the previous part of the question, and apply it to the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best k value that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        classifier = KNeighborsClassifier(n_neighbors=14) # load model w/ best hyperparameters       
        classifier.fit(traindata, trainlabels) # fit the model
        test_pred = classifier.predict(testdata) # make predictions using the test data
        testError = error(testlabels, test_pred) # calculate test error        

        # Do not change this sequence!
        return (classifier, testError)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class Question3(object):
    def LinearSVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15}.

        Write this without using GridSearchCV.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        
        min_err = 1
        for i in range(-5,16):
            C = 2**i
                        
            classifier = LinearSVC(C=C)
            cval_score = cross_val_score(classifier, X=traindata, y=trainlabels, cv=10) # calculate cross val score based on C
            err = 1 - np.mean(cval_score) # cross val error
            
            # update C for the smallest error
            if err < min_err:
                C_min = C
                min_err = err
        
        # Do not change this sequence!
        return (C_min, min_err)

    def SVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15} and \gamma from 2^{-15},...,2^{3}.

        Use GridSearchCV to perform a grid search.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. gamma_min            Float. The hyper-parameter \gamma that minimizes the validation error.
        3. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        
        min_err = 1
        for i in range(-5,16):
            C = 2**i
            for j in range(-15, 4):
                gamma = 2**j
                        
                classifier = SVC(kernel='rbf', C=C, gamma=gamma) 
                cval_score = cross_val_score(classifier, X=traindata, y=trainlabels, cv=10) # calculate cross val score based on C and gamma
                err = 1 - np.mean(cval_score) # cross val error
            
                # update C and gamma for the smallest error
                if err < min_err:
                    C_min = C
                    gamma_min = gamma
                    min_err = err

        # Do not change this sequence!
        return (C_min, gamma_min, min_err)

    def LogisticRegression_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-14},...,2^{14}.

        You may either use GridSearchCV or search by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        
        min_err = 1
        for i in range(-14,15):
            C = 2**i
                        
            classifier = LogisticRegression(C=C)
            cval_score = cross_val_score(classifier, X=traindata, y=trainlabels, cv=10) # calculate cross val score based on C
            err = 1 - np.mean(cval_score) # cross val error
            
            # update C for the smallest error
            if err < min_err:
                C_min = C
                min_err = err

        # Do not change this sequence!
        return (C_min, min_err)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train the best classifier selected above on the whole training set.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best classifier that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        
        classifier = SVC(kernel='rbf', C=8, gamma=0.125) # load model w/ best hyperparameters
        classifier.fit(traindata, trainlabels) # fit model
        test_pred = classifier.predict(testdata) # make predictions using the test data
        testError = error(testlabels, test_pred) # calculate test error
        
        # Do not change this sequence!
        return (classifier, testError)

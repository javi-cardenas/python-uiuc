import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        
        ### formula in textbook on page 19
        ### matrix multiplication - matrix_1 columns must match matrix_2 rows
        
        ### first term = ln(pi)
        first_term = np.log(pi)

        ### second term = means.T * inv_cov * data
#         print(means.shape, np.linalg.inv(cov).shape, data.T.shape)
        second_term = np.dot(np.dot(means, np.linalg.inv(cov)), data.T)
        second_term = second_term.T # transposed for broadcasting

        ### third term = -0.5 * means.T * inv_cov * means
#         print(means.shape, np.linalg.inv(cov).shape, means.T.shape)
        third_term = -0.5*np.dot(np.dot(means, np.linalg.inv(cov)), means.T)
        third_term = np.diag(third_term)

        ### linear objective function = sum of the three terms
        lin_obj_fn = np.add(np.add(first_term, second_term), third_term)

        ### y = argmax(linear objective function)
        y = np.argmax(lin_obj_fn, 1) # vector of labels

#         print('Data:\n', data)
#         print('Priors:\n', pi)
#         print('Means:\n', means)
#         print('Covariance Matrix:\n', cov)        
        
#         print('\nFirst term:\n', first_term.shape)
#         print('Second term:\n', second_term.shape)
#         print('Third term:\n', third_term.shape)
#         print("Lin Obj F'n:\n", lin_obj_fn.shape)
#         print('y:\n', y.shape)

        return y
        
    
    def classifierError(self,truelabels,estimatedlabels):
#         print('True Labels:\n', truelabels.shape)
#         print('Estimated Labels:\n', estimatedlabels.shape)
        
        ### error = number of incorrect labels / total number of labels
        error = np.mean(estimatedlabels != truelabels)
        
        return error


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        trainlabel = trainlabel.astype(int) # cast the array of floats as integers
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist.
        pi = np.zeros(nlabels)            # Store your prior in here
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))   # Store the covariance matrix in here
        # Put your code below
        
        
        ### formulas in textbook on page 20
        
        for i in range(nlabels):
    
            ### calculate the priors
            pi[i] = np.mean(trainlabel == i) # prior for label i
    
            ### calculate the means
            means[i] = np.mean(trainfeat[trainlabel == i], 0) # means for label i
                           
            
        ### calculate the covariance matrix
        for i in range(len(trainfeat)):
            
#             print(trainingdata[i], means[traininglabels[i]])
            xi_minus_mu = trainfeat[i] - means[trainlabel[i]] # for each datapoint subtract mu
#             print(xi_minus_mu)
            cov += np.outer(xi_minus_mu, xi_minus_mu.T) # take the outer product of xi minus mu and its transpose
#             break

        cov = cov / (len(trainfeat) - nlabels) # divide by N - M

#         print('Training Data:', trainfeat.shape)
#         print('Training Labels:', trainlabel.shape)
    
#         print('Priors:\n', pi)
#         print('Means:\n', means)   
#         print('Covariance Matrix:\n', cov)

        # Don't change the output!
        return (pi,means,cov)

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        
        pi, mean, cov = self.trainLDA(trainingdata, traininglabels) # train LDA classifier
        esttrlabels = q1.bayesClassifier(trainingdata, pi, mean, cov) # predicted training labels
        trerror = q1.classifierError(traininglabels, esttrlabels)
        
        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
                
        pi, mean, cov = self.trainLDA(valdata, vallabels) # train LDA classifier        
        estvallabels = q1.bayesClassifier(valdata, pi, mean, cov) # predicted val labels       
        valerror = q1.classifierError(vallabels, estvallabels)
        
        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):          
    
        d_array = dist.cdist(trainfeat, testfeat,"euclidean") # array of distances between each pair of the 2 collection of inputs
        index_array = np.argpartition(d_array, k, 0)[:k] # array of indices that index data along given axis in partitioned order
        labels, count = stats.mode(trainlabel[index_array]) # returns array of modal values and array of counts for each value
        return labels


    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]

        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below   
                       
            k = k_array[i] # choose k
            
            # training            
            esttrlabels = self.kNN(trainingdata, traininglabels, trainingdata, k) # train kNN classifier
            trainingError[i] = q1.classifierError(traininglabels, esttrlabels) # calculate training error
            
            # validation            
            estvallabels = self.kNN(trainingdata, traininglabels, valdata, k) # train kNN classifier
            validationError[i] = q1.classifierError(vallabels, estvallabels) # calculate val error

        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        q1 = Question1()
        
        classifier = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='brute') # adjust model parameters
        start_time = time.time()
        classifier.fit(traindata, trainlabels) # train the model
        fitTime = time.time() - start_time # time taken to fit the model
        
        start_time = time.time()
        estvallabels = classifier.predict(valdata) # predictions based on validation data
        predTime = time.time() - start_time # time taken to make predictions from the validation data    
        valerror = q1.classifierError(vallabels, estvallabels) # calculate val error

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        q1 = Question1()
        
        classifier = LinearDiscriminantAnalysis() # import model
        start_time = time.time()
        classifier.fit(traindata, trainlabels) # train the model
        fitTime = time.time() - start_time # time taken to fit the model
        
        start_time = time.time()
        estvallabels = classifier.predict(valdata) # predictions based on validation data
        predTime = time.time() - start_time # time taken to make predictions from the validation data    
        valerror = q1.classifierError(vallabels, estvallabels) # calculate val error        

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
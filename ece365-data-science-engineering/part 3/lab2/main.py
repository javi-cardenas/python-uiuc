import numpy as np
from sklearn import svm


class Question1(object):
    def svm_classifiers(self, X, y):
        
        # define classifiers
        svm_linear = svm.SVC(kernel='linear')
        svm_non_linear = svm.SVC(kernel='rbf')
        
        # fit classifiers
        svm_linear.fit(X,y)
        svm_non_linear.fit(X,y)
        
        return svm_linear, svm_non_linear

    def acc_prec_recall(self, y_pred, y_test):     
        
        true_pos  = 0
        true_neg  = 0
        false_pos = 0
        false_neg = 0
        
        for i, label_pred in enumerate(y_pred):
            
            if label_pred == 1:
                if label_pred == y_test[i]: # check for true positive
                    true_pos  += 1
                else:
                    false_pos += 1 # check for false positive
                
            elif label_pred == 0:
                if label_pred == y_test[i]: # check for true negative
                    true_neg  += 1
                else:
                    false_neg += 1 # check for false negative
                    
        acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        prec = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        
        return acc, prec, recall


class Question2(object):
    def CCF_1d(self, x1, x2):
        
        d = len(x1)
        ccf = np.zeros_like(x1)
        
        for m in range(d):
            total_sum = 0
            for k in range(d):
                total_sum += x1[k%d] * x2[(m+k)%d] # cross correlation formula    
            ccf[m] = total_sum
        
        return ccf

    def align_1d(self, x1, x2):
        
        x2_copy = x2.copy()
        max_ccf = 0
        
        # shift image until ccf is maximized
        for i in range(len(x1)):       
            ccf = self.CCF_1d(x1,x2_copy)[0]
            
            if ccf > max_ccf:
                max_ccf = ccf
                aligned_sig = x2_copy
                
            x2_copy = np.roll(x2_copy,1)

        return aligned_sig


class Question3(object):
    def CCF_2d(self, x1, x2):
        
        d = len(x1)
        ccf = np.zeros_like(x1)
        
        for m in range(d):
            for n in range(d):
                total_sum = 0
                for m_prime in range(d): 
                    for n_prime in range(d):
                        total_sum += x1[m_prime%d, n_prime%d] * x2[(m_prime+m)%d, (n_prime+n)%d] # cross correlation formula    
                ccf[m,n] = total_sum
            
        return ccf
    
    def align_2d(self, x1, x2):
        
        aligned_img = x2.copy()
        
        m = 1
        n = 1
        while m != 0 and n != 0:
            
            ccf = self.CCF_2d(x1,aligned_img)
            
            # find how much to shift in the x and y directions
            m = np.argmax(ccf, axis=0)[0]
            n = np.argmax(ccf, axis=1)[0]
                        
            # shift in x-direction
            if m != 0:
                aligned_img = np.roll(aligned_img, -m, axis=0)
            
            # shift in y-direction
            if n != 0:
                aligned_img = np.roll(aligned_img, -n, axis=1)
                    
        return aligned_img

    def response_signal(self, ref_images, query_image):
    
        M = ref_images.shape[2]
        resp_signals = np.zeros(M)
          
        for i in range(M):
            ref_image = ref_images[:,:,i]
            ccf       = self.CCF_2d(ref_image, query_image)
            
            norm_ccf  = ccf - np.mean(ccf) # normalized ccf
            resp_sig  = np.max(norm_ccf)   # maximum ccf value
            resp_signals[i] = resp_sig
            
        return resp_signals

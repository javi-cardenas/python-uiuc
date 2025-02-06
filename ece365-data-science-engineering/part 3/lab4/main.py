import numpy as np
from skimage.transform import radon, iradon, resize


class Question1(object):
    def complete_ortho_matrix(self, M):
       
        # each row and col is an orthonormal basis i.e. the sum of a row/col must add up to 1        
        a = M[:,0]
        b = M[:,1]
        c = np.cross(a,b).reshape((-1,1))
        
        orth_mat = np.hstack((M, c))
        
        return orth_mat

    def recover_ortho_matrix(self, M):
        
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        orth_mat = U @ Vt
                             
        return orth_mat

    def comp_rec_ortho_matrix(self, M):
        
        M         = self.complete_ortho_matrix(M)
        orth_mat = self.recover_ortho_matrix(M)

        return orth_mat


class Question2(object):
    def template_matching(self, noisy_proj, I0, M, Tmax):
        
        N = noisy_proj.shape[0]
        proj_angles = np.zeros(N)
        
        phi     = np.array([(2*np.pi*x)/M for x in range(M)]) # all phi_k angles
        phi_deg = phi * (180/(np.pi))                         # converted to degrees for radon function
        
        updated_img = I0.copy()
        
        t = 0
        while t < Tmax:
        
            # update the projection angles
            gammas = radon(updated_img, theta=phi_deg)
                
            for i, proj in enumerate(noisy_proj):
                correlation  = np.inner(proj, gammas) # correlation between template and noisy projection
                L2_norm      = np.linalg.norm(gammas, ord=2) # L2 norm of template
                L = np.argmax(correlation / L2_norm)
                print(correlation.shape)
                print(L2_norm)
                print(L)
                
                proj_angles[i] = phi[L]
                    
            # update the image
            updated_img = iradon(updated_img, theta=proj_angles)
            
            t += 1
            
        theta = proj_angles
        I_rec = updated_img
        
        return I_rec, theta

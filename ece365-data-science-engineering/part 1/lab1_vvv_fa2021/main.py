import numpy as np

class Lab1(object):
    def solver(self, A, b):
        A_inv = np.linalg.inv(A) # inverse matrix of A
        return A_inv.dot(b) # matrix multiplication of A inverse and b

    def fitting(self, x, y):
        ones = np.ones(100) # matrix with all ones
        x_and_ones = np.column_stack((x, ones)) # creates the matrix of x and ones

        pseudo_inv = np.linalg.pinv(x_and_ones) # pseudoinverse matrix
        coeff = pseudo_inv.dot(y) # matrix multiplication of pseudoinverse matrix and y
        return coeff

    def naive5(self, X, A, Y):
        # Calculate the matrix with $(i,j$)-th entry as  $\mathbf{x}_i^\top A \mathbf{y}_j$ by looping over the rows of $X,Y$.
        qf = np.zeros((len(X),len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                qf[i,j] = np.dot(np.dot(X[i],A), Y[j])
        return qf

    def matrix5(self, X, A, Y):
        # Repeat part (a), but using only matrix operations (no loops!).
        return np.dot(np.dot(X,A), Y.T)

    def naive6(self, X, A):
        # Calculate a vector with $i$-th component $\mathbf{x}_i^\top A \mathbf{x}_i$ by looping over the rows of $X$.
        qf = np.zeros(len(X))
        for i in range(len(X)):
            qf[i] = np.dot(np.dot(X[i], A),X[i])
        return qf

    def matrix6(self, X, A):
        # Repeat part (a) using matrix operations (no loops!).
        return np.sum(np.dot(X,A)*X, axis=1)

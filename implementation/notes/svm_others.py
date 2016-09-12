'''
Vicki Niu, July 2k14
Description: Implementation of support vector machine (thanks Vapnik!) in Python
Packages: cvxopt as quadratic solver & numpy as general bringer of joy & mathematical efficiency
'''

import numpy
import cvxopt.solvers


#Trains an SVM
class train(object):
    #Class constructor: kernel function & data
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c
        if self._c is not None: self._c = float(self._c)
        
    #returns trained SVM predictor given features (X) & labels (y)
    def train(self, X, y):
        lagrangeMultipliers = self.multipliers(X, y)
        return self.predictor(X, y, lagrangeMultipliers)
        
    #returns SVM prediction
    def predictor(self, X, y, lagrangeMultipliers):
        svindices = lagrangeMultipliers > 1e-5
        
        svmultipliers = lagrangeMultipliers[svindices]
        sv = X[svindices]
        svlabels = y[svindices]
        
        #compute error assuming zero bias
        bias = numpy.mean(
            [y_i - predict(self._kernel, 0.00, svmultipliers, sv, svlabels).predict(x_i)
            for (y_i, x_i) in zip(svlabels, sv)])
                
        return predict(self._kernel, bias, svmultipliers,sv,svlabels)
        
    #compute Gram matrix
    def gram(self, X):
        n_samples, n_features = X.shape
        K = numpy.zeros((n_samples, n_samples))
        for i in range(0, n_samples):
            for j in range(0, n_samples):
                K[i, j] = self.kernel(X[i], X[j])
        return K
        
    #compute Lagrangian multipliers
    def multipliers(self, X, y):
        n_samples, n_features = X.shape
        K = self.gram(X)
        
        P = cvxopt.matrix(numpy.outer(y,y) * K)
        q = cvxopt.matrix(numpy.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.00)
        
        if self._c is None:
            G = cvxopt.matrix(numpy.diag(numpy.ones(n_samples) * -1))
            h = cvxopt.matrix(numpy.zeros(n_samples))
        else:
            G_1 = cvxopt.matrix(numpy.diag(numpy.ones(n_samples) * -1))
            h_1 = cvxopt.matrix(numpy.zeros(n_samples))
            
            G_2 = cvxopt.matrix(numpy.diag(numpy.ones(n_samples)))
            h_2 = cvxopt.matrix(numpy.ones(n_samples) * self._c)
            
            G = cvxopt.matrix(numpy.vstack(G_1, G_2))
            h = cvxopt.matrix(numpy.vstack(h_1, h_2))
            
        soln = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        return numpy.ravel(soln['x'])

#SVM prediction        
class predict(object):
    #Class constructor
    def __init__(self, kernel, bias, weights, sv, svlabels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._sv = sv
        self._svlabels = svlabels
    
    #Returns SVM predicton given feature vector
    def predict(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights, self._sv, self._svlabels):
            result += z_i * y_i * self._kernel(x_i, x)
            
        return numpy.sign(result).item()
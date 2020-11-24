import numpy as np
import math
from timeit import default_timer as timer

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import MLCourse.utilities as utils

# -------------
# - Baselines -
# -------------

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.0,
            'features': [1,2,3,4,5],
            #'features': list(range(1, 385)),
            


        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        
        numfeatures = Xless.shape[1]
        

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.pinv(inner).dot(Xless.T).dot(ytrain) / numsamples
        

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

# ---------
# - TODO: -
# ---------

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({'regwgt': 0.5}, parameters)


    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
      
        #print(self.params['features'])
        
        
        numfeatures = Xtrain.shape[1]
        #print("NFEATURES!Numsamples ::", Xtrain.shape[1], "N", numfeatures, numsamples)

        inner = (Xtrain.T.dot(Xtrain) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)



        self.weights = np.linalg.pinv(inner).dot(Xtrain.T).dot(ytrain) / numsamples
        #print(self.weights)

    def predict(self, Xtest):
        #Xless = Xtest[:, self.params['features']]
        #print(self.params['features'])
        #Xless = Xtest[:, list(range(1, Xtrain.shape[1]))]
        
        ytest = np.dot(Xtest, self.weights)
        return ytest





class LassoRegression(Regressor):
   
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({'lmda': 0.01}, parameters)


    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]            
        numfeatures = Xtrain.shape[1]
        lmda = self.params['lmda']
        w = np.zeros(numfeatures)
       
        err = np.inf
        tolerance = 0.001
        XX =  (1.0/numsamples)  *  Xtrain.T.dot(Xtrain)
        Xy = (1.0/numsamples)  *  Xtrain.T.dot(ytrain)
        yX = (1.0/numsamples) * ytrain.T.dot(Xtrain)
        yy = (1.0/numsamples) * ytrain.T.dot(ytrain)

        stepsize = 0.5 / np.linalg.norm(XX)

        for i in range(1000):
            c_of_w = w.T.dot(XX).dot(w)   -     w.T.dot(Xy) - yX.dot(w) + yy + (lmda * w.T.dot(w))

            if (abs(c_of_w - err) < tolerance):
                print("broke on iteration, ", i, c_of_w, err)
                break


            err = c_of_w

            #proximal operator:
            new_w = np.subtract(w, (stepsize * XX).dot(w))

            new_w = np.add(new_w, (stepsize * Xy))

 
            for j in range(len(new_w)):
                if (new_w[j] > (stepsize * lmda)):
                    new_w[j] = new_w[j] - (stepsize * lmda)

                elif (abs(new_w[j]) <= (stepsize * lmda)):
                    new_w[j] = 0

                elif new_w[j] < (stepsize * lmda):
                    new_w[j] = new_w[j] + (stepsize * lmda)
                
                           
            w = new_w
        self.weights = w
        

    def predict(self, Xtest):
        
        ytest = np.dot(Xtest, self.weights)
        return ytest






class StochasticGradDescent(Regressor):
    '''
    algorithm 3 from notes:
    for i: 1 .. epochs
      shuffle data points 1, .. n
      for j = 1, .. n:
        g = Grad_cj(w) , where Grad_cj(w) = (X.T.*w -Y_j)x_j
        stepsize_t = i^(-1)
        w = w - (stepsize_t * g)
    return w   

    '''
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({'stepsize': 0.01}, parameters)
        #self.params = utils.update_dictionary_items({'regwgt': 0.5}, parameters)


    def learn(self, Xtrain, ytrain):
        
        numsamples = Xtrain.shape[0]
        numfeatures = Xtrain.shape[1]
        
        stepsize_0 = self.params['stepsize']

        #we declare a new variable, since random.shuffle would otherwise modify Xtrain
        X_train = Xtrain[:]

        n_array = np.arrange(numsamples)

        w = np.random.rand(numfeatures) 
        num_epochs = 1000
        for i in range(1, num_epochs):
            np.random.shuffle(n_array)
            for j in n_array:
               
                gradient = np.subtract(X_train[j].T.dot(w), ytrain[j])
                #if(i == 10): print(gradient)
                #gradient = Xtrain[j] * gradient 
                gradient = np.dot(gradient, (X_train[j]))
                #gradient = gradient * Xtrain[j]

                #I think the question asks for a CONSTANT, small stepsize, not a shrinking one.
                #If not, comment the next line and uncomment the one after it.
                stepsize_t = stepsize_0
                #stepsize_t = stepsize_0  / i 

                
                w = np.subtract(w, (stepsize_t * gradient))

        self.weights = w
       

    def predict(self, Xtest):
        #Xless = Xtest[:, self.params['features']]
        #print(self.params['features'])
        #Xless = Xtest[:, list(range(1, Xtrain.shape[1]))]
       
        ytest = np.dot(Xtest, self.weights)
        return ytest




class BatchGradDescent(Regressor):
    
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({'tau': 0.7}, parameters)


    def learn(self, Xtrain, ytrain):
       
        numsamples = Xtrain.shape[0]       
        numfeatures = Xtrain.shape[1]
        
       
        w = np.random.rand(numfeatures)
        err = np.inf
        tolerance = 0.001
        max_iterations = 1000

        #for 2f, these 2 arrays will hold c_of_w and error, to plot difference between the two for every epoch
        c_of_w_array = []
        err_array = []
        epoch_array = []

        #this array will hold time between each epoch
        time_array = []

        for i in range(max_iterations):
            
            epoch_time = timer()
            time_array.append(epoch_time)

            Xw_minus_y = np.subtract(Xtrain.dot(w), ytrain)                     
            c_of_w = (0.5 / numfeatures) * ((np.linalg.norm(Xw_minus_y, ord=2)**2))
           
            #np.append(c_of_w_array, c_of_w)
            #np.append(err_array, err)
            #np.append(epoch_array, i)
            c_of_w_array.append(c_of_w)
            err_array.append(err)
            epoch_array.append(i)
             
            if (abs(c_of_w - err) < tolerance):
                break
            err = c_of_w
            g = (1.0 /numsamples)  * (Xtrain.T.dot(Xw_minus_y))
            
            

            #linesearch
           
            stepsize_max = 1.0
            tau = self.params['tau']
            stepsize = stepsize_max
            #create new weight parameter, linsearch w, to not modify original w during line search
            lsw = w

            obj = c_of_w
            ls_max_itterations = 100

            for j in range(ls_max_itterations):
                lsw = np.subtract(w, (stepsize * g))

                lswXw_minus_y = np.subtract(Xtrain.dot(lsw), ytrain)                     
                c_of_lsw = (0.5 / numfeatures) * ((np.linalg.norm(lswXw_minus_y, ord=2)**2))
                

                if ( c_of_lsw < (obj - tolerance)):
                    break
                
                stepsize = stepsize * tau
                if (j + 1) == ls_max_itterations:
                    #could not improve solution
                    
                    stepsize = 0
                    
            
                
            #-------------
            
            w = w - (stepsize * g)
            
            



        #c_of_w_array
        #err_array
        x = np.subtract(c_of_w_array, err_array)
        x = np.absolute(x)
        
        #uncomment to plot X: epochs, Y: error
        #plt.plot(epoch_array, x)
        #plt.show()
        #epoch_array 
        
        #uncomment to plot X: time, Y: error
        #plt.plot(time_array, x)
        #plt.show()

        #to find out how many epochs the optimization took, uncomment this next line
        #print(epoch_array[-1])
        self.weights = w
       

    def predict(self, Xtest):
        #Xless = Xtest[:, self.params['features']]
        #print(self.params['features'])
        #Xless = Xtest[:, list(range(1, Xtrain.shape[1]))]
        #print(Xtest.shape[1])
        ytest = np.dot(Xtest, self.weights)
        return ytest

#!/usr/bin/env python
# coding: utf-8

# In[54]:


#Spencer Tu & Kenneth Kitahata
#02/20/2020
#HW1b: STAT 339: HOMEWORK 1B (LINEAR REGRESSION)
import scipy
import numpy as np
from random import seed
from random import randint
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn import preprocessing

womens100data = np.loadtxt(fname = "http://colindawson.net/data/womens100.csv", delimiter = (","))

# 1a)
#linear regression function that takes Nx2 matrix of ordered pairs (t, x) 
#returns 2x1 array OLS coefficients (w0, w1)
def regression(data):
    #hardcode 1's in the first column
    n = len(data)
    x = np.sum(data[:, 0]) / n
    t = np.sum(data[:, 1]) / n
    xt = np.dot(data[:, 0], data[:, 1]) / n
    x_sq = np.sum(data[:, 0] ** 2) / n
    w1 = (xt - (x * t)) / (x_sq - (x ** 2))
    w0 = t - (w1 * x)
    return np.array([w0, w1])

coefourmodel = regression(womens100data)    
print("Our model:", coefourmodel)

# 1b)
# helper function to plot a line from slope and intercept
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

plt.scatter(womens100data[:, 0], womens100data[:, 1])
abline(coefourmodel[1], coefourmodel[0])
plt.savefig('HW1b_image1.png')

#compare performance to Scikit regression model 
from sklearn.linear_model import LinearRegression
x = womens100data[:, 0]
x = x.reshape(-1, 1)
y = womens100data[:, 1]

reg = LinearRegression().fit(x, y)
coef2 = reg.coef_ 
coef2 = np.concatenate(([reg.intercept_], coef2))
print("Package reg model:", coef2)

#The constant and coefficient are the same
plt.scatter(womens100data[:, 0], womens100data[:, 1])
abline(coef2[1], coef2[0])


# In[55]:


#1c)
regression(womens100data)
time2012 = regression(womens100data)[0] + 2012*regression(womens100data)[1]
print("2012time:", time2012)
time2012act = 10.75
squaredpredictionerror2012 = (time2012 - time2012act)**2
time2016 = regression(womens100data)[0] + 2016*regression(womens100data)[1]
print("2016time:", time2016)
time2016act = 10.71
squaredpredictionerror2016 = (time2016 - time2016act)**2


print("squared prediction error 2012:", squaredpredictionerror2012)
print("squared prediction error 2016:", squaredpredictionerror2016)


# In[56]:


#1d & 1h)
#take in two matrixes => xmatrix is a matrix of independent variables
# y or "t" on the side is a matrix (1xN) of dependent variable
#return an array of OLS estimators => w0, w1, w2, w3, w4.....
def regwithmultiplex(xmatrix, t, lamda = None):
    #all the comments are based on the data I made from women100data (look one cell above)
    #t is 19x1
    #xmatrix is 19x2      
    ones = np.ones((len(xmatrix),1))
    
    #now xmatrix is 19x3
    xmatrix = np.concatenate((ones, xmatrix), axis =1)
    
    #xmatrix_trans is 3x19
    xmatrix_trans = np.transpose(xmatrix)
    
    #xtranspose_dot_x is 3x3
    xtranspose_dot_x = np.dot(xmatrix_trans,xmatrix)

    #new stuff
    if lamda:
        identity = np.identity(len(xtranspose_dot_x))     #creating an identity matrix with (len(xtranspose_dot_x)) rows
        lamda_i = lamda*identity
        xtranspose_dot_x = np.add(xtranspose_dot_x, lamda_i)
    #new stuff end
    
    # inverse is 3x3
    inverse = np.linalg.inv(xtranspose_dot_x)

    #x_transpose_t is 3x1
    x_transpose_t = np.dot(xmatrix_trans, t)
    
    OLScoef = np.dot(inverse, x_transpose_t)
    
    return OLScoef
    


# In[57]:


#test, do not delete, this is a test data set to run regwithmultiplex (1d)
#"one" is a 19x3 array. Testing 1d to see if it works for a matrix
x = np.zeros((len(womens100data),1))  #making a 19x1 array
y = np.ones((len(womens100data),1))    #making a 19x1 array
for i in range(len(womens100data)):
    x[i] = womens100data[i][0]
    y[i] = womens100data[i][1]
    
random = np.zeros((len(womens100data),1))
seed(1)
for i in range(len(womens100data)):
    random[i] = randint(1,50)

one = np.concatenate((random, x), axis =1)

regwithmultiplex(one, y)


# In[58]:


#testing 1d with 
regwithmultiplex(x, y)


# In[59]:


#1e)
#take in single col, int D, and return matrix of predictors to order D

def polynomial(data, D):   
    #ones = np.ones((len(data), 1))                       # I comment this out since I already append
    #data = np.concatenate((ones, data), axis = 1)        #a column of 1s in regwithmultiplex()
    
    for i in range(2, D + 1):    
        data_x_power = data[:, 0] ** i                  #Ken - I changed 1 to 0 
        data_x_power = np.reshape(data_x_power, (-1, 1))
        data = np.concatenate((data, data_x_power), axis = 1)
    return data


# In[60]:


#1f)
#returns the OLS estimators (coefficients)
#takes in target (y values), a predictor column (x values), and a positive int for order of polynomials
#will return an array of OLS estimators => w0 (constant), w1(coef for x^0), w2 (coef for x^1), w3 (coef for x^2)...
def OLScoef(target, predictor, D, lamda = None):
    return regwithmultiplex(polynomial(predictor, D), target, lamda)


# In[61]:


#1g)
#plot a graph of D=3 on synthdata2016
synthdata2016 = np.loadtxt(fname = "http://colindawson.net/data/synthdata2016.csv", delimiter = (","))
synthdata2016x = np.zeros((len(synthdata2016),1))  
synthdata2016y = np.ones((len(synthdata2016),1)) 

for i in range(len(synthdata2016)):                          #seperating synthdata2016 into two columns
    synthdata2016x[i] = synthdata2016[i][0]                  # one for x (predictor), one for y (target)   
    synthdata2016y[i] = synthdata2016[i][1]

#the fourth argument is lamda
coef = OLScoef(synthdata2016y, synthdata2016x, 3)
print(coef)

#plotting the graph (the dots)
plt.scatter(synthdata2016x, synthdata2016y)

#trying to get the line to show
xforline = np.zeros((len(synthdata2016x),1))
for i in range (len(xforline)):
    xforline[i] = synthdata2016x[i]
xforline = np.sort(xforline, axis = 0)

yforline = np.zeros((len(synthdata2016y),1))
for i in range(len(yforline)):
    yforline[i] = coef[0] + coef[1]*xforline[i] + coef[2]*(xforline[i]**2) + coef[3]*(xforline[i]**3) 
 
plt.plot(xforline, yforline)


# In[62]:


#1h is with 1d
#compare results of different lambdas: 10^1, 10^2, ... , 10^5
for j in range(1, 6):    
    coef = OLScoef(synthdata2016y, synthdata2016x, 3, 10 ** j)

    #plotting the graph (the dots)
    plt.scatter(synthdata2016x, synthdata2016y)

    #trying to get the line to show
    xforline = np.zeros((len(synthdata2016x),1))
    for i in range (len(xforline)):
        xforline[i] = synthdata2016x[i]
    xforline = np.sort(xforline, axis = 0)

    yforline = np.zeros((len(synthdata2016y),1))
    for i in range(len(yforline)):
        yforline[i] = coef[0] + coef[1]*xforline[i] + coef[2]*(xforline[i]**2) + coef[3]*(xforline[i]**3) 
        
    #label each line 1 through j    
    plt.plot(xforline, yforline, label = 'line ' + str(j))
    
plt.legend()


# In[63]:


#1i)
#returns a FUNCTION that when called on a new predictor matrix returns a vector of predictions
#new predictor's should be N*D+1 dimension
#the new predictor should already be in the correct form: column1 is Xn^1, column2 is Xn^2.....
#if it's not, uncomment 4 lines below.

def OLSfunction (target, predictor, D, lamda = None):
    coefs = OLScoef(target, predictor, D, lamda)
    def vector_of_predictions(new_predictor):
        #new_predictor = polynomial(new_predictor, D)
        predictions = np.zeros((len(new_predictor),1))
        for i in range(len(new_predictor)):
            for j in range(len(new_predictor[0])):
                #print(len(new_predictor[0]))
                #print(i)
                #print(j)
                #print(coefs[j+1])
                #print(new_predictor[i,j])
                predictions[i] = (coefs[j+1]*(new_predictor[i,j])) + predictions[i]
                #print(predictions[i])
            predictions[i] = predictions[i] + coefs[0]
            #print(predictions[i])
            #print("end")
        return predictions
    return vector_of_predictions

        


# In[64]:


#2a)
#function that takes target vector, prediction vector, returns mean squared error (MSE)

def MSE(target, prediction):
        sq_error = (target - prediction) ** 2
        MSE = np.sum(sq_error) / len(target)
        return MSE
    
#MSE(synthdata2016y, predict_synth)


# In[65]:


#2b) cross validation
#returns N x 2 where 1st col is fold number and 2nd col is MSE
#returns mean error, std dev, mean error for training set, std. dev 

def CV(target, prediction, k = 10, seed = 1, D = 1, lamda = 0, report_training = False):
    #create one dataset (prediction + targets) to shuffle
    data = np.append(target, prediction, axis = 1)
    data = preprocessing.scale(data)
    np.random.seed(seed)
    np.random.shuffle(data)
    
    #separate targets and predictors
    data_y = data[:, 0]
    data_x = data[:, 1:]

    #split data into k folds
    data_x = np.array_split(data_x, k)
    data_y = np.array_split(data_y, k)
    length = list(range(k))
    mse_array = ([])
    
    #iterate through each fold 
    for i in length:        
        validate_x = data_x[i]
        validate_y = data_y[i]
        index = length.copy()
        del index[i]

        #for first index, set training data equal to first array
        for j in index:
            if j == index[0]:
                train_x = data_x[j]
                train_y = data_y[j]
            
            #append each additional array to training data
            else:
                train_x = np.append(train_x, data_x[j], axis = 0)                                
                train_y = np.append(train_y, data_y[j], axis = 0)
                
        #fit model based on training data for i-th fold
        model_fit = OLSfunction(train_y, train_x, D, lamda)
        predict_coef = model_fit(validate_x)  
        
        #calculate training error for each fold, append to trainingerror_array
        if report_training == True:
            training_predict_coef = model_fit(train_x)
            trainingerror = MSE(train_y, training_predict_coef)
            trainingerror = np.array([trainingerror])
            if i == length[0]:
                trainingerror_array = trainingerror
            else:
                trainingerror_array = np.append(trainingerror_array, trainingerror, axis = 0)                                
        
        #calculate MSE for each fold, append to mse_array
        mse = MSE(validate_y, predict_coef)
        mse_array = np.append(mse_array, mse)   
    
    mse_array = np.reshape(mse_array, (-1, 1))
    mean_error = np.sum(mse_array) / k
    std_dev = np.std(mse_array)
    results = np.array([mean_error, std_dev])
    
    if report_training == True:    
        mean_train_error = np.sum(trainingerror_array) / k
        std_dev_train_error = np.std(trainingerror_array)
        results = np.append(results, [mean_train_error, std_dev_train_error])

    results = np.reshape(results, (-1,1))
    return results
    
mse = CV(y, x, 10, 1, 2, 0, False)
mse


# In[66]:


#2c)
#a function to select the best polynomial order using cross-validation
#predictor is single column
#A positive integer D, representing the maximum order of polynomial to consider 
def bestpolynomialorder(target, predict, D = 1, k = 2, seed = 1, lamda = 0, report_training = False):
    
    for i in range(1, D + 1):                          
        if i == 1:
            mse = CV(target, predict, k, seed, i, lamda, report_training)
            mse_polynomial = np.array(mse)
        else:
            mse_append = CV(target, predict, k, seed, i, lamda, report_training)
            mse_append = np.array(mse_append)
            mse_polynomial = np.append(mse_polynomial, mse_append, axis = 1)
            
    #finding the order with the smallest MSE
   
    min_mse = np.argmin(mse_polynomial[0, :])
    return np.array([(min_mse + 1), mse_polynomial[0, min_mse]])

bestpolynomialorder(y, x, D = 10, k = 10, seed = 1)


# In[67]:


#2d)
#apply model to synth data / womens data
womens100data = np.loadtxt(fname = "http://colindawson.net/data/womens100.csv", delimiter = (","))
x = np.reshape(womens100data[:, 0], (-1, 1))
x = x.reshape(-1, 1)
y = np.reshape(womens100data[:, 1], (-1, 1))


synthdata2016 = np.loadtxt(fname = "http://colindawson.net/data/synthdata2016.csv", delimiter = (","))
synthdata2016x = np.reshape(synthdata2016[:, 0], (-1, 1))  
synthdata2016y = np.reshape(synthdata2016[:, 1], (-1, 1)) 

print(bestpolynomialorder(y, x, D = 20, k = 10, seed = 1))
#found optimal polynomial is 1 for k = 10
print(bestpolynomialorder(y, x, D = 20, k = 19, seed = 1))
#found optimal polynomial is 1 for k = N


# In[68]:


print(bestpolynomialorder(synthdata2016y, synthdata2016x, D = 10, k = 10, seed = 1))
#found optimal polynomial is 2 for k = 10
print(bestpolynomialorder(synthdata2016y, synthdata2016x, D = 10, k = 50, seed = 1))
#found optimal polynomial is 2 for k = N


# In[69]:


#2e)
womens100datafor2e = np.loadtxt(fname = "http://colindawson.net/data/womens100.csv", delimiter = (","))
womens100datafor2e
CV(y, x, k = 10, seed = 1, D = 19, lamda = 0, report_training = False)


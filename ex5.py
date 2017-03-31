import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import csv
from matplotlib import cm

def logistic_z(z): 
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x): 
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1
#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features
def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(xrange(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in xrange(dim):
            update_grad = 1 ### something needs to be done here
            w[i] = w[i] + learn_rate ### something needs to be done here
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        for i in xrange(dim):
            update_grad=0.0
            for n in xrange(num_n):
                update_grad+=(-logistic_wx(w,x_train[n])+y_train[n])# something needs to be done here
            w[i] = w[i] + learn_rate * update_grad/num_n
    return w


def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in xrange(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm)
    print "error=",np.mean(error)
    return w

def read_file(file):
    data = np.genfromtxt(file, delimiter='\t', dtype=float)
    return data

def L_simple_function(w):
    L = (logistic_wx(w, [1,0])-1)**2+(logistic_wx(w, [0,1]))**2+(logistic_wx(w, [1,1])-1)**2
    return L

def log_der(w, x):
    x1 = x[0]
    x2 = x[1]
    dw1 = (x1*np.exp(np.inner(w,x)))/((1+np.exp(np.inner(w,x)))**2)
    dw2 = (x2*np.exp(np.inner(w,x)))/((1+np.exp(np.inner(w,x)))**2)
    return [dw1, dw2]

def l_simple_der(w):
    dl_simple_w1 = 2*(logistic_wx(w,[1,0])-1)*log_der(w,[1,0])[0] + 2*logistic_wx(w,[0,1])*log_der(w,[0,1])[0] + 2*(logistic_wx(w,[1,1])-1)*log_der(w,[1,1])[0]
    dl_simple_w2 = 2*(logistic_wx(w,[1,0])-1)*log_der(w,[1,0])[1] + 2*logistic_wx(w,[0,1])*log_der(w,[0,1])[1] + 2*(logistic_wx(w,[1,1])-1)*log_der(w,[1,1])[1]

    return [dl_simple_w1, dl_simple_w2]

def gradient_descent(dim, learning_rate, n_iter=1000):
    w = np.random.rand(dim)
    for i in range(2):
        for n in range(n_iter):
            w[i] = w[i] -learning_rate*l_simple_der(w)[i]
    return w


def main():
    test = read_file('data/data_small_nonsep_test.csv')

    # Task 1.1
    x = np.arange(-6,6,0.1)
    y = np.arange(-6,6,0.1)
    X, Y = np.meshgrid(x, y)
    print X.shape
    Z=np.zeros(X.shape)
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            Z[i,j]=L_simple_function((X[i,j],Y[i,j]))
    print np.argmin(Z)
    title = 'L_simple, min= ' + str(round(np.min(Z),5)) 
    plt.pcolormesh(X,Y,Z)
    plt.colorbar()
    plt.title(title)
    #plt.show()

    #Task 1.2

    learning_rate = [0.0001, 0.01, 0.1, 1, 10, 100]
    w_n = [0]*len(learning_rate)
    l_simple_n = [0]*len(learning_rate)

    #fig = plt.figure()
    for i in range(len(learning_rate)):
        w_n[i] = gradient_descent(2, learning_rate[i]) #dimension of w=2, w1 and w2
        l_simple_n[i] = L_simple_function(w_n[i])
        print(w_n[i],l_simple_n[i])
        # ax = fig.add_subplot(3,2,i+1)
        # print l_simple_n[i]
        # ax.plot(l_simple_n[i])
        # title = 'L_rate= ' + str(learning_rate[i])
        # ax.set_title(title)

    plt.figure()
    plt.plot(learning_rate,l_simple_n, )
    plt.show()


    
main()

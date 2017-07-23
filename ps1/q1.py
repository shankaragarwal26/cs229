import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def newton_method():
    X = read_X()
    Y = read_Y()
    number_of_features = len(X[0])
    theta = np.zeros(number_of_features)

    iterations = 200
    count =1
    while True:
        grad = np.matrix(get_gradient(theta,X,Y))
        H = np.matrix(get_hessian_inverse(theta,X,Y))
        value = vector_multiplication(H,np.transpose(grad))
        arr = convert_matrix_to_array(value)
        theta =np.subtract(theta,arr)
        if count>iterations:
            break
        count +=1

    return theta


def read_Y():
    Y = np.loadtxt('logistic_y.txt')
    for i in range(0,len(Y)):
        if Y[i] == -1:
            Y[i]=0
    return Y


def read_X():
    temp = np.loadtxt('logistic_x.txt')
    features = len(temp[0])
    X = np.zeros((len(temp),features+1))
    for i in range(0,len(temp)):
        X[i][0] = 1
        for j in range(0,features):
            X[i][j+1] = temp[i][j]
    return X


def convert_matrix_to_array(value):
    length = len(value)
    arr = np.zeros(length)
    for i in range(0, length):
        arr[i] = value[i, 0]
    return arr


def vector_multiplication(v1,v2):
    v3 = np.dot(v1,v2)
    return v3


def hypothesis(theta,X):
    hyp = np.dot(np.matrix(theta),np.matrix(X).transpose())
    hyp = np.array(np.transpose(hyp))
    for i in range(0,len(hyp)):
        hyp[i] = 1.0/(1.0 + np.exp(-1.0* hyp[i]))

    return hyp



def get_hessian_inverse(theta,X,Y):
    hessian = np.zeros((len(theta),len(theta)))
    for k in range(0,len(X)):
        for i in range(0,len(theta)):
            for j in range(0,len(theta)):
                hessian[i][j] += (-1*X[k][i]*X[k][j])
    return inv(np.matrix(hessian))


def get_gradient(theta,X,Y):
    length = len(X)
    grad = np.zeros(len(theta))
    hyp = hypothesis(theta,X)
    for i in range(0,length):
        for j in range(len(grad)):
            grad[j] += (Y[i] - hyp[i])*X[i][j]
    return grad


def get_mod_value(theta):
    length = len(theta)
    print(theta)
    value = 0
    for i in range(0,length):
        value += theta[i]*theta[i]
    return np.sqrt(value)


def get_colors_from_y(Y):
    colors = []
    for i in range(0,len(Y)):
        if Y[i] == -1:
            colors.append('red')
        else:
            colors.append('blue')
    return colors


def plot_x_y():
    X = np.loadtxt('logistic_x.txt')
    Y = np.loadtxt('logistic_y.txt')
    color = get_colors_from_y(Y)
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[:,0],X[:,1],c=color)
    plt.show()


def get_X2_theta(X, theta):
    res = np.zeros(len(X))
    for i in range(0,len(X)):
        res[i] = (-1.0 * (theta[0]*X[i][0] + theta[1]*X[i][1]))/theta[2]
    return res




def plot_x_y_with_hyperplane(theta):
    X = read_X()
    Y = np.loadtxt('logistic_y.txt')
    color = get_colors_from_y(Y)
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(X[:,1],X[:,2],c=color)
    res = get_X2_theta(X,theta)
    plt.plot(X[:,1],res,c='magenta')

    plt.show()


if __name__ == "__main__":
    theta = newton_method()
    plot_x_y()
    plot_x_y_with_hyperplane(theta)
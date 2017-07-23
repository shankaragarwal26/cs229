import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from numpy.linalg import inv
from numpy.linalg import norm

def weighted_linear_regression(lbl,tau):

    X, Y, train_float = get_input()
    Y = Y[0].transpose()
    tau = 2 * tau * tau

    result = []

    length = len(X)

    for data in X:
        W = np.zeros((length,length))

        for i in range(0,length):
            ith_value = X[i,1]
            current_value = data[1]
            W[i,i] = np.exp((-1.0 * np.square(ith_value-current_value))/tau)

        first_term = inv(np.dot(np.dot(np.transpose(X), W), X))
        second_term = np.dot(np.dot(np.transpose(X),W),Y)
        theta = np.dot(first_term,second_term)

        value = np.dot(np.transpose(theta),data)

        result.append(value[0,0])
    print(result)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(train_float[0], train_float[1], c='r', label='raw-data')
    ax1.set_xlim([1150, 1600])
    ax1.set_ylim([-2.0, 8.0])
    tau_5 = plt.plot(X[:,1],result,c='magenta',label=lbl)
    plt.legend([tau_5],[lbl])

    plt.savefig(lbl+".png")


def get_input():
    train_float = genfromtxt('quasar_train.csv', delimiter=',')
    lambda_value = np.matrix(train_float[0]).transpose()
    quasar_train = np.matrix(train_float[1:])
    X = np.zeros((len(lambda_value), 2))
    for i in range(0, len(lambda_value)):
        X[i, 0] = 1
        X[i, 1] = lambda_value[i, 0]
    return X, quasar_train, train_float


def get_input_test():
    train_float = genfromtxt('quasar_test.csv', delimiter=',')
    lambda_value = np.matrix(train_float[0]).transpose()
    quasar_train = np.matrix(train_float[1:])
    X = np.zeros((len(lambda_value), 2))
    for i in range(0, len(lambda_value)):
        X[i, 0] = 1
        X[i, 1] = lambda_value[i, 0]
    return X, quasar_train, train_float


def linear_regression():
    X, Y, train_float = get_input()
    Y = Y[0].transpose()
    first = inv(np.dot(np.transpose(X),X))
    second = np.dot(np.transpose(X),Y)
    theta = np.dot(first,second)
    print(theta)


def weighted_linear_regression_row(X ,Y):

    tau = 5
    tau = 2 * tau * tau

    result = []

    length = len(X)

    for data in X:
        W = np.zeros((length,length))

        for i in range(0,length):
            ith_value = X[i,1]
            current_value = data[1]
            W[i,i] = np.exp((-1.0 * np.square(ith_value-current_value))/tau)

        first_term = inv(np.dot(np.dot(np.transpose(X), W), X))
        second_term = np.dot(np.dot(np.transpose(X),W),Y)
        theta = np.dot(first_term,second_term)

        value = np.dot(np.transpose(theta),data)

        result.append(value[0,0])
    return result


def append_array_to_matrix(smooth_values, new_y, row):
    for i in range(0,len(new_y)):
        smooth_values[row][i] = new_y[i]
    return smooth_values


def maximum(param1, param2):
    if param1 > param2:
        return param1
    return param2


def answer_c():
    X, Y, train_float = get_input()
    smooth_train = np.zeros((len(Y),len(X)))
    for i in range(0,len(Y)):
        train_y = Y[i].transpose()
        new_y = weighted_linear_regression_row(X, train_y)
        smooth_train = append_array_to_matrix(smooth_train,new_y,i)

    X_test,Y_test,test_float = get_input_test()
    smooth_test = np.zeros((len(Y_test), len(X_test)))
    for i in range(0,len(Y_test)):
        test_y = Y_test[i].transpose()
        new_y = weighted_linear_regression_row(X_test, test_y)
        smooth_test = append_array_to_matrix(smooth_test,new_y,i)

    right_trains = smooth_train[:,150:]
    left_trains = smooth_train[:,:50]
    right_tests = smooth_test[:,150:]
    left_tests = smooth_test[:,:50]

    m = len(right_trains)

    # dist_matrix = np.zeros((m, m))
    # max_value = 1
    # for i in range(0,m):
    #     for j in range(0,m):
    #         val = np.square(right_trains[i,:] - right_trains[j,:])
    #         val = np.sum(val)
    #         dist_matrix[i,j] = val
    #         max_value = maximum(max_value,val)
    #
    # f_left_estimates_train = np.zeros((len(left_trains),50))
    # k = 3
    #
    # for i in range(0,m):
    #     distance = dist_matrix[i,:]
    #     h = np.max(distance)
    #     distance = np.divide(distance,h)
    #     ind = np.argsort(distance,axis=0)
    #     close_index = np.zeros((m,1))
    #     for j in range(0,k):
    #         index = ind[j]
    #         close_index[index] =1
    #
    #     kerns = np.zeros(len(distance))
    #
    #     for j in range(0,len(kerns)):
    #         kerns[j] = maximum((1-distance[j])/h, 0)
    #
    #     for j in range(0, len(close_index)):
    #         kerns[j] = kerns[j] * close_index[j]
    #     sum = np.sum(kerns)
    #     kerns = np.divide(kerns,sum)
    #     # Got 3 distances
    #     value = np.multiply(np.transpose(left_trains),kerns)
    #     value = np.sum(value,axis=1)
    #     f_left_estimates_train[i,:] = value
    #
    # # Calculate Average Error
    #
    # sum = np.sum(np.square(left_trains-f_left_estimates_train))
    # err = sum /m
    # print("Training Error " + str(err))


    ## Test Error Calculations
    test_length = len(right_tests)

    dist_matrix = np.zeros((test_length, test_length))
    max_value = 1
    for i in range(0,test_length):
        for j in range(0,test_length):
            val = np.square(right_tests[i,:] - right_tests[j,:])
            val = np.sum(val)
            dist_matrix[i,j] = val
            max_value = maximum(max_value,val)

    f_left_estimates_test = np.zeros((test_length,50))
    k = 3

    for i in range(0,test_length):
        distance = dist_matrix[i,:]
        h = np.max(distance)
        distance = np.divide(distance,h)
        ind = np.argsort(distance,axis=0)
        close_index = np.zeros((test_length,1))
        for j in range(0,k):
            index = ind[j]
            close_index[index] =1

        kerns = np.zeros(len(distance))

        for j in range(0,len(kerns)):
            kerns[j] = maximum((1-distance[j])/h, 0)

        for j in range(0, len(close_index)):
            kerns[j] = kerns[j] * close_index[j]
        sum = np.sum(kerns)
        kerns = np.divide(kerns,sum)

        print(np.shape(kerns))
        print(np.shape(left_tests))
        # Got 3 distances
        value = np.multiply(np.transpose(left_tests),kerns)
        value = np.sum(value,axis=1)
        f_left_estimates_test[i,:] = value

    # Calculate Average Error

    sum = np.sum(np.square(left_tests-f_left_estimates_test))
    err = sum /m
    print("Testing Error " + str(err))




if __name__ == "__main__":
    # lbl = "tau_5"
    # weighted_linear_regression(lbl,5)
    #
    # lbl = "tau_1"
    # weighted_linear_regression(lbl,1)
    #
    # lbl = "tau_10"
    # weighted_linear_regression(lbl,10)
    #
    # lbl = "tau_100"
    # weighted_linear_regression(lbl,100)
    #
    # lbl = "tau_1000"
    # weighted_linear_regression(lbl,1000)
    # linear_regression()

    answer_c()
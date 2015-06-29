"""
plot statistics
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_dict():

    path="numpy_training_cost.txt"
    content_numpy_cost = np.asarray([float(line.strip().split(':')[1]) for line in open(path,'rb')])
    path="numpy_training_time.txt"
    content_numpy_time = np.asarray([float(line.strip().split(':')[1]) for line in open(path,'rb')])

    path="slice_training_cost.txt"
    content_slice_cost = np.asarray([float(line.strip().split(':')[1]) for line in open(path,'rb')])
    path="slice_training_time.txt"
    content_slice_time = np.asarray([float(line.strip().split(':')[1]) for line in open(path,'rb')])

    numpy_cost=[]
    numpy_time=[]
    for cost, time in zip(content_numpy_cost, content_numpy_time):
        if len(numpy_cost)==0 or cost < numpy_cost[-1]:
            numpy_cost.append(cost)
            numpy_time.append(time)

    slice_cost=[]
    slice_time=[]
    for cost, time in zip(content_slice_cost, content_slice_time):
        if len(slice_cost)==0 or cost < slice_cost[-1]:
            slice_cost.append(cost)
            slice_time.append(time)

    # do in term of accuracy for a better look
    for i in xrange(len(numpy_cost)):
        numpy_cost[i] = 1 - numpy_cost[i]
    for i in xrange(len(slice_cost)):
        slice_cost[i] = 1 - slice_cost[i]

    #from scipy.interpolate import interp1d
    #xnew = np.linspace(0, 10, 40)

    f0, = plt.plot( numpy_cost[2:], numpy_time[2:])
    plt.hold(True)
    f1, = plt.plot( slice_cost[2:], slice_time[2:])

    plt.show()

if __name__ == '__main__':

    plot_dict()

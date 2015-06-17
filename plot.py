"""
plot statistics
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_dict(L_dict):

    mean_time_numpy =[]
    var_time_numpy = []
    mean_time_slice =[]
    var_time_slice = []
  

    for d in L_dict:
        for key in d :
            mean_time_numpy.append( np.mean(d[key]['numpy']) )
            mean_time_slice.append( np.mean(d[key]['slice']))
            var_time_numpy.append( np.mean(d[key]['numpy']) + np.std(d[key]['numpy']) )
            var_time_slice.append( np.mean(d[key]['slice']) + np.std(d[key]['slice']) )

    f0, = plt.plot(mean_time_numpy)

    plt.hold(True)
    f1, = plt.plot(mean_time_slice)
    plt.show()


if __name__ == '__main__':
    paths= ["/data/lisatmp2/ducoffem/stats_fuel_0",
            "/data/lisatmp2/ducoffem/stats_fuel_0_64",
            "/data/lisatmp2/ducoffem/stats_fuel_0_128",
            "/data/lisatmp2/ducoffem/stats_fuel_0_1024"]

    L_dict = []
    for path in paths:
        L_dict.append(pickle.load(open(path, 'r')))

    plot_dict(L_dict)


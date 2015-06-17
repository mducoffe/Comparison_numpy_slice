"""
Comparison between aessing data using numpy indexing and slice
Statistics for a small dataset
"""
import numpy
import pickle
from contextlib import closing
import time

def test_small_dataset(nb_elem, size_elem, size_minibatch, access, dtype='float32', dict_result={}):
    """
    Parameters :
    ----------------
    nb_elem : number of elements in the database (int)
    size_elem : size of a sample in the database (tuple)
    size_minibatch : number of elements in a minibatch (int)
    access : how to access the data : str in {'random', 'sequential'}
    dtype :
    """
    assert access in ['random', 'sequential']
    # no in load memory
    # generate an artifical database
    database = numpy.random.ranf((nb_elem,) +size_elem)
    database = database.astype(dtype)
    t_access_numpy = []
    t_access_slice = []
    for i in xrange(500):
        if access == 'sequential':
            for j in xrange(nb_elem/size_minibatch):
                index = numpy.arange(j*size_minibatch,(j+1)*size_minibatch)
                start = time.clock()
                elem = database[index]
                end = time.clock()
                t_access_numpy.append(end - start)      
        else:
            raise Exception('unplemented option %s',access)

    # a tort ou a raison on serait tenter de ne pas faire l'appel successif pour ne pas que l'acces soit plus aise car en memoire ?
    for i in xrange(500):
        if access == 'sequential':
            for j in xrange(nb_elem/size_minibatch):
                slice_obj = slice(j*size_minibatch, (j+1)*size_minibatch)
                start = time.clock()
                elem = database[slice_obj]
                end = time.clock()
                t_access_slice.append(end - start)      
        else:
            raise Exception('unplemented option %s',access)

    dict_result['database_'+str(nb_elem)+"_"+str(numpy.prod(size_elem))] = {'numpy': t_access_numpy, 'slice': t_access_slice}
    return dict_result


if __name__ == '__main__':
    dict_ = {}
    nb_elem = range(1000, 10000, 1000)
    size_elem = (28, 28)
    size_minibatch = 1024
    access = 'sequential'
    saving_path = "/data/lisatmp2/ducoffem/stats_fuel_0_1024"
    for nb_elem_ in nb_elem:
        dict_ = test_small_dataset(nb_elem_, size_elem, size_minibatch, access)
        f = open(saving_path,'w')
        pickle.dump(dict_, f, -1)
        f.close()
        
    

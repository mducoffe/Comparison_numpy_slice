import six
from picklable_itertools.base import BaseItertool
from picklable_itertools.extras import partition_all
from picklable_itertools import imap, iter_, islice
from fuel.schemes import BatchScheme

class partition_slice(BaseItertool):

    def __init__(self, n, seq):
        self._n = n
        self._seq = iter( self.create_slice(seq)) # not picklable but who cares ?

    def create_slice(self, seq):
        items = []
        for p in xrange(len(seq)/self._n):
            items.append(slice(p*self._n, (p+1)*self._n))
        return items


    def __next__(self):
        items = []
        try:
            #or _ in six.moves.xrange(self._n):
            items.append(next(self._seq))
        except StopIteration:
            pass
        if len(items) == 0:
            raise StopIteration
        return items[0]


class SequentialScheme_slice(BatchScheme):
    """Sequential batches iterator.
    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.
    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.
    """
    def get_request_iterator(self):
        return partition_slice(self.batch_size, self.indices)


if __name__ == '__main__':
    import numpy as np
    indices = np.arange(100)
    batch_size = 10
    it_1 = partition_slice(batch_size, indices)
    it_2 = partition_all(batch_size, indices)
    print indices[it_1.next()]
    print indices[list(it_2.next())]
    print indices[it_1.next()]
    print indices[list(it_2.next())]



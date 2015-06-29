# extension for balancing
import time
import blocks
from blocks.extensions import SimpleExtension

# this is an old implementation that will be replaced
class Time_reference(SimpleExtension):

    def __init__(self, start, index_time, **kwargs):
        # `params_dict` contains both the parameters and the momentums
        super(Time_reference, self).__init__( **kwargs)

        self.start = start
        self.index_time = index_time

    def do(self, which_callback, *args):

        self.index_time.set_value( time.clock() - self.start)

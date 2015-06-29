# maxout classifier
import numpy as np
import theano
import theano.tensor as T
from blocks.initialization import Constant, Uniform
from blocks.bricks.base import application

#from batch_normalize import ConvolutionalLayer, ConvolutionalActivation, Linear
# change for cpu tests

floatX = theano.config.floatX
from blocks.bricks.conv import Flattener
from blocks.bricks import Initializable, Sequence

class Classifier(Initializable):


    def __init__(self,
                    conv_seq, fully_connected,
                    **kwargs):
        super(Classifier, self).__init__(**kwargs)

        self.conv_seq = conv_seq
        self.flatten = Flattener()
        self.fully_connected = fully_connected
        
        self.children = [self.conv_seq, self.flatten, self.fully_connected]

    def get_dim(self, name):
        if name=="input":
            return self.conv_seq.get_dim(name)
        elif name == 'output':
            return self.fully_connected.get_dim(name)
        else:
            super(Classifier, self).get_dim(name)

    @application(inputs=['input_'], outputs=['output_'])
    def apply(self, input_):
        output_conv = self.conv_seq.apply(input_)
        input_fully = self.flatten.apply(output_conv)
        output = self.fully_connected.apply(input_fully)
        output.name = "output_"
        return output

"""
test : comparaison d'un entrainement effectue avec numpy ou slice
"""
import time
import os, sys
import getopt
import numpy as np
import theano.tensor as T
from classifier import Classifier
from iterator_slice import SequentialScheme_slice
from extension import Time_reference
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence, ConvolutionalActivation, Flattener
from blocks.bricks import Linear, Rectifier, MLP, Identity, Softmax
from blocks.initialization import Constant, Uniform, IsotropicGaussian
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.utils import shared_floatx
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.bricks.cost import MisclassificationRate
from blocks.algorithms import Momentum, RMSProp, Scale, GradientDescent
from blocks.graph import ComputationGraph

from fuel.datasets import MNIST



def build_model_mnist():

    # CNN
    filter_size = (5, 5)
    activation = Rectifier().apply
    pooling_size = (2, 2)
    num_filters = 50
    layer0 = ConvolutionalLayer(activation=activation, filter_size=filter_size, num_filters=num_filters,
                              pooling_size=pooling_size,
                              weights_init=Uniform(width=0.1),
                              biases_init=Uniform(width=0.01), name="layer_0")

    filter_size = (3, 3)
    activation = Rectifier().apply
    num_filters = 20
    layer1 = ConvolutionalLayer(activation=activation, filter_size=filter_size, num_filters=num_filters,
                              pooling_size=pooling_size,
                              weights_init=Uniform(width=0.1),
                              biases_init=Uniform(width=0.01), name="layer_1")

    conv_layers = [layer0, layer1]
    convnet = ConvolutionalSequence(conv_layers, num_channels= 1,
                                    image_size=(28, 28))

    convnet.initialize()
    output_dim = np.prod(convnet.get_dim('output'))
    mlp = MLP(activations=[Identity()], dims=[output_dim, 10],
                        weights_init=Uniform(width=0.1),
                        biases_init=Uniform(width=0.01), name="layer_2")
    mlp.initialize()

    classifier = Classifier(convnet, mlp)
    classifier.initialize()
    return classifier

def training_model_mnist(learning_rate, momentum, iteration, batch_size, epoch_end, iter_batch):

    x = T.tensor4('features')
    y = T.imatrix('targets')

    classifier = build_model_mnist()

    predict = classifier.apply(x)
    y_hat = Softmax().apply(predict)

    cost = Softmax().categorical_cross_entropy(y.flatten(), predict)
    cost.name = "cost"
    cg = ComputationGraph(cost)
    error_brick = MisclassificationRate()
    error_rate = error_brick.apply(y.flatten(), y_hat)
    error_rate.name = "error"


    train_set = MNIST(('train', ))
    test_set = MNIST(("test",))

    if iteration =="slice":
        data_stream = DataStream.default_stream(
                train_set, iteration_scheme=SequentialScheme_slice(train_set.num_examples,
                                                            batch_size))
        data_stream_test = DataStream.default_stream(
                test_set, iteration_scheme=SequentialScheme_slice(test_set.num_examples,
                                                            batch_size))
    else:
        data_stream = DataStream.default_stream(
                train_set, iteration_scheme=SequentialScheme(train_set.num_examples,
                                                            batch_size))

        data_stream_test = DataStream.default_stream(
                test_set, iteration_scheme=SequentialScheme(test_set.num_examples,
                                                            batch_size))

    step_rule = Momentum(learning_rate=learning_rate,
                         momentum=momentum)

    start = time.clock()
    time_spent = shared_floatx(np.float32(0.), name="time_spent")
    time_extension = Time_reference(start, time_spent, every_n_batches=1)

    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    monitor_train = TrainingDataMonitoring(
        variables=[cost], prefix="train", every_n_epochs=iter_batch)
    monitor_valid = DataStreamMonitoring(
        variables=[cost, error_rate, time_spent], data_stream=data_stream_test, prefix="valid",
        every_n_epochs=iter_batch)

    # add a monitor variable about the time
    extensions = [  monitor_train,
                    monitor_valid,
                    FinishAfter(after_n_epochs=epoch_end),
                    Printing(every_n_epochs=iter_batch),
                    time_extension
                  ]

    main_loop = MainLoop(data_stream=data_stream,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)
    main_loop.run()


def usage():
    print "THEANO_FLAGS=\"device=gpu,floatX=float32\" python test_mnist_hdf5 --learning_rate=0.1 --momentum=0.9 --iteration={slice, numpy} --batch_size=32 --epoch_end=40 --iter_epoch=1"


def main(argv):

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["learning_rate=", "momentum=", "iteration=", "batch_size=", "epoch_end=", "iter_batch="])

    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    learning_rate = 0.1
    momentum = 0.9
    iteration = "slice"
    batch_size = 4
    epoch_end = 40
    iter_batch = 1

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--learning_rate"):
            learning_rate = float(a)
        elif o in ("--momentum"):
            momentum = float(a)
        elif o in ("--batch_size"):
            batch_size = int(a)
        elif o in ("--epoch_end"):
            epoch_end = int(a)
        elif o in ("--iter_batch"):
            iter_batch = int(a)
        elif o in ("--iteration"):
            assert a.lower() in ['numpy', 'slice']
            iteration = a
        else:
            assert False, "unhandled option"

    training_model_mnist(learning_rate=learning_rate, momentum=momentum, iteration=iteration,
                         batch_size=batch_size, epoch_end=epoch_end, iter_batch=iter_batch)

if __name__ == "__main__":
    main(sys.argv)

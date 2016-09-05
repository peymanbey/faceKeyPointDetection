from utils import data_set, shared_dataset, build_update_functions, early_stop_train
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import identity, rectify, leaky_rectify
import theano.tensor as T

def build_model_vanila_CNN(X, stride=1):

    net = {}
    non_linear_function = leaky_rectify
    net['input'] = InputLayer((None, 1,96,96), input_var=X)
    net['fc1'] = DenseLayer(net['input'], num_units=512, nonlinearity=non_linear_function)

    # net['fc1']  = ConvLayer(incoming=net['input'],
    #                         num_filters=1024,
    #                         filter_size=1,
    #                         stride=1,
    #                         pad=0,
    #                         nonlinearity=non_linear_function,
    #                         flip_filters=False)
    # net['fc1_dropout'] = DropoutLayer(net['fc1'], p=0)
    net['prob'] = DenseLayer(net['fc1'], num_units=30, nonlinearity=identity)

    return net

if __name__ == "__main__":

    # path to train and testing data
    PATH_train = "../data/training.csv"
    PATH_test = "../data/test.csv"

    # load data
    print 'loading data'
    data = data_set(path_train=PATH_train, path_test=PATH_test)

    # data.augment()
    # center data
    print 'center alexnet'
    data.center_alexnet()
    # print 'center Xs VGG Style, X doesnt have missing values'
    # data.center_VGG()

    # generate test validation split
    train_set_x = data.X
    valid_set_x = data.X_val
    train_set_y = data.y
    valid_set_y = data.y_val

    print 'shape of train X', train_set_x.shape, 'and y', train_set_y.shape
    print 'shape of validation X', valid_set_x.shape, 'and y', valid_set_y.shape

    # build the mask matrix for missing values, load it into theano shared variable
    # build masks where 0 values correspond to nan values
    temp = np.isnan(train_set_y)
    train_MASK = np.ones(temp.shape)
    train_MASK[temp] = 0
    # still have to replace nan with something to avoid propagation in theano
    train_set_y[temp] = -100
    temp = np.isnan(valid_set_y)
    val_MASK = np.ones(temp.shape)
    val_MASK[temp] = 0
    # still have to replace nan with something to avoid propagation in theano
    valid_set_y[temp] = -100

    # load into theano shared variable
    print 'load data to gpu'
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    val_MASK, train_MASK = shared_dataset(val_MASK, train_MASK)


    X = T.ftensor4('X')
    y = T.matrix('y')

    batch_size = 512
    l2 = .0000001

    net = build_model_vanila_CNN(X=X, stride=1  )
    network = net['prob']

    train_fn, val_fn = build_update_functions(train_set_x=train_set_x, train_set_y=train_set_y,
                                              valid_set_x=valid_set_x,valid_set_y= valid_set_y,
                                              y= y,X= X,network=network,
                                              val_MASK=val_MASK, train_MASK=train_MASK,
                                              l2_reg=l2,
                                              batch_size=batch_size)
    print 'compile done successfully'

    # call early_stop_train function
    early_stop_train(train_set_x, train_set_y,
                     valid_set_x, valid_set_y,
                     network, train_fn, val_fn,
                     batch_size=batch_size)
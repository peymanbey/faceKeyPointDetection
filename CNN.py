from utils import data_set, shared_dataset, build_update_functions, early_stop_train
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer, get_all_layers
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import identity, rectify
import theano.tensor as T
import cPickle as pickle


def single_conv_layer(input_layer, **kwargs):

    complex_layer = ConvLayer(incoming=input_layer,**kwargs)
    # complex_layer = PoolLayer(complex_layer, pool_size=2, stride=2, mode='average_exc_pad')

    return complex_layer


def build_model_vanila_CNN(X, channel = 1,stride=1):
    # TODO: set according to daniels guide
    conv1filters = 64
    conv2filters = 64
    conv3filters = 128
    conv4filters = 256

    net = {}
    non_linear_function = rectify


    net['input'] = InputLayer((None, channel, 96, 96), input_var=X)

    net['conv1'] = single_conv_layer(net['input'],
                               num_filters=conv1filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=non_linear_function,
                               flip_filters=False)

    net['conv2'] = single_conv_layer(net['conv1'],
                               num_filters=conv2filters,
                               filter_size=2,
                               stride=stride,
                               pad=1,
                               nonlinearity=non_linear_function,
                               flip_filters=False)

    net['conv2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, mode='average_exc_pad')


    net['conv3'] = single_conv_layer(net['conv2'],
                                     num_filters=conv3filters,
                                     filter_size=2,
                                     stride=stride,
                                     pad=1,
                                     nonlinearity=non_linear_function,
                                     flip_filters=False)
    #
    net['conv4'] = single_conv_layer(net['conv3'],
                                     num_filters=conv4filters,
                                     filter_size=3,
                                     stride=stride,
                                     pad=1,
                                     nonlinearity=non_linear_function,
                                     flip_filters=False)
    net['conv4'] = PoolLayer(net['conv4'], pool_size=2, stride=2, mode='average_exc_pad')
    # net['fc5'] = DenseLayer(net['conv4'], num_units=512, nonlinearity=non_linear_function)

    net['fc5']  = ConvLayer(incoming=net['conv4'],
                            num_filters=500,
                            filter_size=1,
                            stride=1,
                            pad=0,
                            nonlinearity=non_linear_function,
                            flip_filters=False)
    # net['fc5'] = DropoutLayer(net['fc5'], p=0.5)
    # net['fc5'] = ConvLayer(incoming=net['fc5'],
    #                        num_filters=500,
    #                        filter_size=1,
    #                        stride=1,
    #                        pad=0,
    #                        nonlinearity=non_linear_function,
    #                        flip_filters=False)
    # net['fc5'] = DropoutLayer(net['fc5'], p=0.3)

    net['fc6'] = DenseLayer(net['fc5'], num_units=30, nonlinearity=identity)

    net['prob'] = NonlinearityLayer(net['fc6'], nonlinearity=identity)

    return net

if __name__ == "__main__":

    # path to train and testing data
    PATH_train = "../data/training.csv"
    PATH_test = "../data/test.csv"

    # load data
    print 'loading data \n'
    data = data_set(path_train=PATH_train, path_test=PATH_test)

    print 'sobel stacking image'
    data.stack_origi_sobel()

    # augmentation
    # data.augment()

    # center data
    # print 'center alexnet \n'
    # data.center_alexnet()
    # print 'center Xs VGG Style, X doesnt have missing values \n'
    # data.center_VGG()


    # generate test validation split
    data.split_trainval()
    train_set_x = data.X
    valid_set_x = data.X_val
    train_set_y = data.y
    valid_set_y = data.y_val
    n_ch = train_set_x.shape[1]

    print 'shape of train X', train_set_x.shape, 'and y', train_set_y.shape,'\n'
    print 'shape of validation X', valid_set_x.shape, 'and y', valid_set_y.shape, '\n'

    # build the mask matrix for missing values, load it into theano shared variable
    # build masks where 0 values correspond to nan values
    temp = np.isnan(train_set_y)
    train_MASK = np.ones(temp.shape)
    train_MASK[temp] = 0
    # still have to replace nan with something to avoid propagation in theano
    train_set_y[temp] = -1000
    temp = np.isnan(valid_set_y)
    val_MASK = np.ones(temp.shape)
    val_MASK[temp] = 0
    # still have to replace nan with something to avoid propagation in theano
    valid_set_y[temp] = -1000

    # load into theano shared variable
    print 'load data to gpu \n'
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    val_MASK, train_MASK = shared_dataset(val_MASK, train_MASK)


    X = T.ftensor4('X')
    y = T.matrix('y')

    batch_size = 16
    l2 = .0001

    #####################################################
    # # Continue a previous run
    # with open("results_backup.p", "rb") as f:
    #     best_network_params, best_val_loss_, best_epoch_,train_loss_history_, val_loss_history_, network = pickle.load(f)
    # # extract input var
    # print 'extract input var \n'
    # X = get_all_layers(network)[0].input_var
    #####################################################
    #  New run
    net = build_model_vanila_CNN(X=X, channel= n_ch, stride=1  )
    network = net['prob']
    #####################################################

    train_fn, val_fn = build_update_functions(train_set_x=train_set_x, train_set_y=train_set_y,
                                              valid_set_x=valid_set_x,valid_set_y= valid_set_y,
                                              y= y,X= X,network=network,
                                              val_MASK=val_MASK, train_MASK=train_MASK,
                                              batch_size=batch_size,l2_reg=l2)
    print 'compile done successfully \n'

    # call early_stop_train function
    early_stop_train(train_set_x, train_set_y,
                     valid_set_x, valid_set_y,
                     network, train_fn, val_fn,
                     batch_size=batch_size)
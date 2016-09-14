
import numpy as np
import theano
import lasagne
from lasagne.objectives import squared_error, aggregate
from lasagne.regularization import regularize_layer_params, l2,regularize_layer_params_weighted
from lasagne.updates import rmsprop
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, get_output, get_all_params, get_all_layers
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
import pickle

from utils import data_set, shared_dataset,  early_stop_train
import theano.tensor as T


def build_model_VGG(weights, X):
    net = {}

    ################################
    ################################
    # net['input'] = InputLayer((None, 3, 224, 224))
    # net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False,
    #                          W=weights[0], b=weights[1])
    net['input'] = InputLayer((None, 1, 96, 96), input_var=X)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=96,
                             filter_size=3,
                             stride=1,
                             pad=1,
                             flip_filters=False)
    ################################

    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001, )  # caffe has alpha = alpha * pool_size

    ################################
    ################################
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=2, ignore_border=False)
    ################################

    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False,
                             W=weights[2], b=weights[3])
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False,
                             W=weights[4], b=weights[5])
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False,
                             W=weights[6], b=weights[7])
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False,
                             W=weights[8], b=weights[9])

    ################################
    ################################
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=2, ignore_border=False)
    ################################

    ################################
    ################################
    # net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    # net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['pool5'], num_units=1024)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.125)
    net['fc8'] = DenseLayer(net['drop7'], num_units=30, nonlinearity=lasagne.nonlinearities.identity)

    return net



def build_update_functions(train_set_x, train_set_y,
                           valid_set_x, valid_set_y,
                           network,
                           y, X,
                           train_MASK, val_MASK,
                           batch_size=32,
                           l2_reg=.0001):

    # build update functions
    # extract tensor representing the network predictions
    prediction = get_output(network)

    ###################New#########################
    # Aggregate the element-wise error into a scalar value using a mask
    # note that y should note contain NAN, replace them with 0 or -1. The value does not matter. It
    # is not used to calculate the aggregated error and update of the network.
    # MASK should be a matrix of size(y), with 0s in place of NaN values and 1s everywhere else.

    # build tensor variable for mask
    trainMASK = T.matrix('trainMASK')
    # collect squared error
    loss_RMSE = squared_error(prediction, y)
    # Drop nan values and average over the remaining values
    loss_RMSE = aggregate(loss_RMSE, weights=trainMASK, mode='normalized_sum')
    # compute the square root
    loss_RMSE = loss_RMSE.sqrt()
    ###############################################

    # add l2 regularization
    # l2_penalty = regularize_layer_params(network, l2)
    # regc = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    # layers = get_all_layers(network)
    # reg_weights = {key: value for (key, value) in zip(layers, regc)}
    # l2_penalty = regularize_layer_params_weighted(reg_weights, l2)

    loss = loss_RMSE#(1 - l2_reg) * loss_RMSE + l2_reg * l2_penalty

    # get network params
    params = get_all_params(network)
    # subset_params = params
    #subset network params to extract the ones that you want to train
    # print 'length of params',len(params), '\n'
    subset_params = [params[0], params[1], params[10], params[11], params[12], params[13]]


    #
    print('RMSPROP \n')
    updates = rmsprop(loss, subset_params, learning_rate=1e-4)
    # create validation/test loss expression
    # the loss represents the loss for all the labels
    test_prediction = get_output(network, deterministic=True)

    ###################New#########################
    # Aggregate the element-wise error into a scalar value using a mask
    # note that y should note contain NAN, replace them with 0 or -1. The value does not matter. It
    # is not used to calculate the aggregated error and update of the network.
    # MASK should be a matrix of size(y), with 0s in place of NaN values and 1s everywhere else.


    # build tensor variable for mask
    valMASK = T.matrix('valMASK')
    # collect squared error
    test_loss = squared_error(test_prediction, y)
    # Drop nan values and average over the remaining values
    test_loss = aggregate(test_loss, weights=valMASK, mode='normalized_sum')
    # compute the square root
    test_loss = test_loss.sqrt()
    ################################################
    # index for mini-batch slicing
    index = T.lscalar()

    # training function
    train_set_x_size = train_set_x.get_value().shape[0]
    val_set_x_size = valid_set_x.get_value().shape[0]

    train_fn = theano.function(inputs=[index],
                               outputs=[loss, loss_RMSE],
                               updates=updates,
                               givens={X: train_set_x[
                                          index * batch_size: T.minimum((index + 1) * batch_size, train_set_x_size)],
                                       y: train_set_y[
                                          index * batch_size: T.minimum((index + 1) * batch_size, train_set_x_size)],
                                       trainMASK: train_MASK[index * batch_size: T.minimum((index + 1) * batch_size,
                                                                                           train_set_x_size)]})
    # validation function
    val_fn = theano.function(inputs=[index],
                             outputs=[test_loss, prediction],
                             givens={X: valid_set_x[
                                        index * batch_size: T.minimum((index + 1) * batch_size, val_set_x_size)],
                                     y: valid_set_y[
                                        index * batch_size: T.minimum((index + 1) * batch_size, val_set_x_size)],
                                     valMASK: val_MASK[
                                              index * batch_size: T.minimum((index + 1) * batch_size, val_set_x_size)]})
    return train_fn, val_fn


if __name__ == "__main__":

    X = T.ftensor4('X')
    y = T.matrix('y')

    batch_size = 32
    l2_val = 0

    ########################################
    ########################################
    ########################################

    # load and set model params
    model = pickle.load(open('../lasagne_modelzoo/vgg_cnn_s.pkl'))
    CLASSES = model['synset words']
    MEAN_IMAGE = model['mean image']
    print MEAN_IMAGE.shape
    # build the model
    net = build_model_VGG(model['values'], X=X)
    network = net['fc8']

    ########################################
    ########################################
    ########################################
    # path to train and testing data
    PATH_train = "../data/training.csv"
    PATH_test = "../data/test.csv"

    # load data
    print 'loading data \n'
    data = data_set(path_train=PATH_train, path_test=PATH_test)

    # print 'sobel image edges'
    # data.sobel_image()

    # augmentation
    # data.augment()

    # center data
    print 'center alexnet \n'
    data.center_alexnet()
    # print 'center Xs VGG Style, X doesnt have missing values \n'
    # data.center_VGG()


    # generate test validation split
    train_set_x = data.X
    valid_set_x = data.X_val
    train_set_y = data.y
    valid_set_y = data.y_val

    print 'shape of train X', train_set_x.shape, 'and y', train_set_y.shape, '\n'
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

    train_fn, val_fn = build_update_functions(train_set_x=train_set_x, train_set_y=train_set_y,
                                              valid_set_x=valid_set_x, valid_set_y=valid_set_y,
                                              y=y, X=X, network=network,
                                              val_MASK=val_MASK, train_MASK=train_MASK,
                                              batch_size=batch_size, l2_reg=l2_val)

    print 'compile done successfully \n'

    # call early_stop_train function
    early_stop_train(train_set_x, train_set_y,
                     valid_set_x, valid_set_y,
                     network, train_fn, val_fn,
                     batch_size=batch_size)
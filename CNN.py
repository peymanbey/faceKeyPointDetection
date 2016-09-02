from utils import data_set, shared_dataset, reinitiate_set_params
import time
from sklearn.cross_validation import train_test_split
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import  get_output,get_all_params,get_all_param_values, set_all_param_values
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import rectify, identity
from lasagne.objectives import squared_error
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import rmsprop, adagrad, nesterov_momentum
import cPickle as pickle

def build_update_functions(train_set_x,train_set_y,
                           valid_set_x,valid_set_y,
                           network,
                           y,X,
                           batch_size = 32,
                           l2_reg = .01,
                           learning_rate = .005,
                           momentum = .9):
    # build update functions
    #####################################
    # extract tensor representing the network predictions
    prediction = get_output(network)
    loss_RMSE = 0
    # collect squared error
    loss_RMSE = squared_error(prediction, y)
    # compute the root mean squared errror
    loss_RMSE = loss_RMSE.mean().sqrt()
    # add l2 regularization
    l2_penalty = regularize_network_params(network,l2)
    loss = (1-l2_reg) * loss_RMSE + l2_reg * l2_penalty
    # get network params
    params = get_all_params(network)
    
#     # create update criterion    
#     print('nestrov')
#     updates = nesterov_momentum(
#         loss, params, learning_rate=learning_rate, momentum=momentum)
    
#     print('AdaGrad')
#     updates = adagrad(loss, params,learning_rate= 1e-3)
    
    print('RMSPROP')
    updates = rmsprop(loss, params, learning_rate= 1e-3)
    # create validation/test loss expression
    # the loss represents the loss for all the lables
    test_prediction = get_output(network,
                                                deterministic=True)
    # collect squared error
    test_loss = squared_error(test_prediction,
                                                 y)
    # compute the root mean squared errror
    test_loss = test_loss.mean().sqrt()
#     test_loss_withl2 = (1-l2_reg) * test_loss + l2_reg * l2_penalty
    
    # index for minibatch slicing
    index = T.lscalar()
    
    # training function 
    train_set_x_size = train_set_x.get_value().shape[0]
    val_set_x_size = valid_set_x.get_value().shape[0]
    
    train_fn = theano.function(inputs=[index],
                               outputs=[loss,loss_RMSE],
                               updates=updates,
                               givens={X:train_set_x[index * batch_size: T.minimum((index + 1) * batch_size,train_set_x_size)],
                                       y:train_set_y[index * batch_size: T.minimum((index + 1) * batch_size,train_set_x_size)]})
    # validation function 
    val_fn = theano.function(inputs=[index],
                             outputs=[test_loss,prediction],
                             givens={X:valid_set_x[index * batch_size: T.minimum((index + 1) * batch_size,val_set_x_size)],
                                     y:valid_set_y[index * batch_size: T.minimum((index + 1) * batch_size,val_set_x_size)]})
    return train_fn,val_fn


def build_model_vanila_CNN(X, stride=1):

    conv1filters = 64
    conv2filters = 64
    conv3filters = 128
    conv4filters = 64
    net = {}
    net['input'] = InputLayer((None, 1, 96, 96), input_var=X)

    net['conv1_1'] = ConvLayer(incoming=net['input'],
                               num_filters=conv1filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=rectify,
                               flip_filters=False)

    net['conv1_2'] = ConvLayer(incoming=net['conv1_1'],
                               num_filters=conv1filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=rectify,
                               flip_filters=False)

    net['pool1'] = PoolLayer(net['conv1_2'], pool_size=2, stride=2)

    net['conv2_1'] = ConvLayer(incoming=net['pool1'],
                               num_filters=conv2filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=rectify,
                               flip_filters=False)

    net['conv2_2'] = ConvLayer(incoming=net['conv2_1'],
                               num_filters=conv2filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=rectify,
                               flip_filters=False)

    net['pool2'] = PoolLayer(net['conv2_2'], pool_size=2, stride=2)

    net['conv3_1'] = ConvLayer(incoming=net['pool2'],
                               num_filters=conv3filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=rectify,
                               flip_filters=False)


    net['conv3_2'] = ConvLayer(incoming=net['conv3_1'],
                               num_filters=conv3filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=rectify,
                               flip_filters=False)

    net['pool3'] = PoolLayer(net['conv3_2'], pool_size=2, stride=2)

    net['conv4_1'] = ConvLayer(incoming=net['pool3'],
                               num_filters=conv4filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=rectify,
                               flip_filters=False)


    net['conv4_2'] = ConvLayer(incoming=net['conv4_1'],
                               num_filters=conv4filters,
                               filter_size=3,
                               stride=stride,
                               pad=1,
                               nonlinearity=rectify,
                               flip_filters=False)

    net['pool4'] = PoolLayer(net['conv4_2'], pool_size=2, stride=2)

    net['fc3'] = DenseLayer(net['pool4'], num_units=256)
    net['fc4'] = DenseLayer(net['fc3'], num_units=30)
    net['prob'] = NonlinearityLayer(net['fc4'], nonlinearity=identity)
    return net

def early_stop_train(train_set_x,train_set_y,
                     valid_set_x,valid_set_y,
                     network,train_fn,val_fn):
    """Get the network and update functions as input and apply early stop training.
    Should return a trained network with training history.
    ----------------------
    Input
    ----------------------
    train_set_x: Training samples, loaded to GPU by theano.shared()
    valid_set_x: Test samples, loaded to GPU by theano.shared()
    train_set_y: Training outputs, loaded to GPU by theano.shared()
    valid_set_y: Training outputs, loaded to GPU by theano.shared()
    network: Deep model, the output layer of the network build using lasagne
    train_fn: theano.function to update the network
    val_fn: theano.function to calculate validation loss
    ----------------------
    Outputs
    ----------------------
    train_loss_history
    val_loss_history_
    network
    ----------------------
    """
    # network parameters
    n_iter = 20000
    improvement_threshold = 0.995
    patience = 20000
    batch_size = 128
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size + 1
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size + 1
    patience_increase = 2
    validation_frequency = min(n_train_batches, patience // 10)
    train_loss_history_temp = []
    best_val_loss_ = np.inf
    best_epoch_ = 0
    epoch = 0
    done_looping = False
    train_loss_history_ = []
    val_loss_history_ = []

    print 'start training'
    start_time = time.time()
    while (epoch < n_iter) and (not done_looping):
        epoch += 1

         # go over mini-batches for a full epoch
        for minibatch_index in range(n_train_batches):

            # update network for one mini-batch
            minibatch_average_cost, minibatch_average_RMSE = train_fn(minibatch_index)

            # store training loss of mini-batches till the next validation step
            train_loss_history_temp.append(minibatch_average_RMSE)

            # number of mini-batches checked
            num_minibatch_checked = (epoch - 1) * n_train_batches + minibatch_index

            # if validation interval reached
            if (num_minibatch_checked + 1) % validation_frequency == 0:

                # compute validation loss
                validation_losses = [val_fn(i)[0] for i in range(n_valid_batches)]

                # store mean validation loss for validation set
                current_val_loss = np.mean(validation_losses)

                # store training and validation history
                train_loss_history_.append(np.mean(train_loss_history_temp))
                val_loss_history_.append(current_val_loss)
                train_loss_history_temp = []

                # is it the best validation loss so far?
                if current_val_loss < best_val_loss_:

                    # increase patience if improvement is significant
                    if (current_val_loss < best_val_loss_ * improvement_threshold):
                        patience = max(patience, num_minibatch_checked * patience_increase)

                    # save the-so-far-best validation RMSE and epoch and model-params
                    best_val_loss_ = current_val_loss
                    best_epoch_ = epoch
                    best_network_params = get_all_param_values(network)
                    # TODO: save the best model as pickle file
                    pickle.dump([best_network_params,best_val_loss_,best_epoch_],
                                open("results.p", "wb"))

            # check if patience exceeded and set the training loop to stop
            if (patience <= num_minibatch_checked):
                print 'patience reached'
                # reset the network weights to the best params saved
                print 'reseting the network params to that of the best seen'
                reinitiate_set_params(network=network,
                                      weights=best_network_params)
                # done optimising, break the optimisation loop
                done_looping = True
                break

        freq = 10
        if (epoch % freq) == 0:
            print (('epoch %i, minibatch %i/%i, validation loss of %f, patience %i, in %f secs') %
                   (epoch, minibatch_index + 1, n_train_batches, current_val_loss, patience, time.time() - start_time))
            start_time = time.time()

    return 0

if __name__ == "__main__":
    # path to train and testing data
    PATH_train = "../data/training.csv"
    PATH_test = "../data/test.csv"
    # load data
    print 'loading data'
    data = data_set(path_train=PATH_train, path_test=PATH_test)
    #  drop the missing values
    print 'drop missing values'
    data.drop_missing_values()
    # center data VGG style
    # print 'center alexnet'
    # data.center_alexnet()
    print 'center VGG'
    data.center_VGG()
    # generate test validation split
    train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(
        data.X, data.y, test_size=0.2, random_state=42)
    # change type and load to GPU
    print 'load data to gpu'
    train_set_x = train_set_x.reshape(-1, 1, 96, 96).astype(theano.config.floatX)
    valid_set_x = valid_set_x.reshape(-1, 1, 96, 96).astype(theano.config.floatX)
    train_set_y = train_set_y.astype(theano.config.floatX)
    valid_set_y = valid_set_y.astype(theano.config.floatX)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)

    X = T.ftensor4('X')
    y = T.matrix('y')
    net = build_model_vanila_CNN(X, stride=1   )
    network = net['prob']
    train_fn, val_fn = build_update_functions(train_set_x, train_set_y,
                                              valid_set_x, valid_set_y,
                                              network,
                                              y, X)
    print 'compile done successfully'

    # call early_stop_train function
    early_stop_train(train_set_x, train_set_y,
                     valid_set_x, valid_set_y,
                     network, train_fn, val_fn)
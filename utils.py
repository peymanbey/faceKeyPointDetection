from os.path import expanduser
import time
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from lasagne.objectives import squared_error, aggregate
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import rmsprop, adagrad, nesterov_momentum
from lasagne.layers import  get_output,get_all_params,set_all_param_values, get_all_param_values
from lasagne.init import GlorotUniform
import cPickle as pickle
from skimage.filters import sobel, sobel_h, sobel_v
from skimage.exposure import equalize_hist


class data_set(object):
    """

    """
    def __init__(self,
                 path_train,
                 path_test):
        """
        Get the path to the training and testing data and store them
        """
        self.PATH_train = path_train
        self.PATH_test = path_test
        self._load_data()

    def _load_data(self):

        """
        Load the training data according to self.PATH_train
        Extract X,y and store in self.X_nan,self.y_nan,including nan values
        Calculate the alexnet-like mean image and store it in self.meanImageAlex
        Calculate the VGG-like mean value, per channel mean, store it in
        self.meanImageVGG
        """
        self.df = pd.read_csv(expanduser(self.PATH_train))
        self.df['Image'] = self.df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
        # tell the user that there are missing values
        print '{} samples from the total of {} have missing values \n'.format(
            self.df.isnull().any(axis=1).sum(), self.df.shape[0])
        print 'Missing values appear in {} different columns of output targets \n'.format(
            self.df.isnull().any(axis=0).sum())
        # extract X,y
        self._extract_Xy()
        print 'shape of X', self.X.shape, 'and y', self.y.shape


    def _extract_Xy(self, col = None):

        if not col:
            col = 'Image'
        # extract X and y
        self.X = np.vstack(self.df[col].values).astype(np.float32)
        self.y = self.df[self.df.columns[:-1]].values.astype(np.float32)

    def split_trainval(self):
        # Shuffle the data
        self.X, self.y = shuffle(self.X, self.y, random_state=47)

        #######################################
        # # scale inputs
        # # TODO scaling to test the guide
        # self.X = self.X / 255.
        # self.y = (self.y - 48) / 48
        # TODO zero mean images, 0meanind
        # temp = self.X.T - self.X.mean(axis = 1)
        # self.X = temp.T
        ######################################
        # reshape
        self.X = self.X.reshape(self.X.shape[0], -1, 96, 96).astype(theano.config.floatX)
        self.y = self.y.astype(theano.config.floatX)

        # train validation split
        self.X, self.X_val, self.y, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=55)

        # Calculate mean image. Note that X doesnt have missing values
        self.meanImageAlex = self.X.mean(axis=0)
        # calculate mean value pre channel, here ve only have one channel
        self.meanImageVGG = self.X.mean()

    def augment(self):
        # augment training data only applies to training set
        print 'augmenting the training data \n'
        tempX = np.copy(self.X)
        tempX =  tempX[:, :, :, ::-1]
        tempy = np.copy(self.y)
        tempy[:,::2] = 96 - tempy[:,::2]
        self.X = np.concatenate((self.X, tempX), axis=0)
        self.y = np.concatenate((self.y, tempy), axis=0)

        # Shuffle the data
        self.X, self.y = shuffle(self.X, self.y, random_state=47)

        # Calculate mean image. Note that X doesnt have missing values
        self.meanImageAlex = self.X.mean(axis=0)
        # calculate mean value pre channel, here ve only have one channel
        self.meanImageVGG = self.X.mean()


    def drop_missing_values(self):
        """"
        Drop the samples that contain missing values.
        The effect of running this function is irreversible.
        """
        # Drop samples with missing values
        self.df = self.df.dropna()
        # extract X and y
        self._extract_Xy()
        print 'shape of X',self.X.shape, 'and y',self.y.shape

    def center_alexnet(self, X=None):

        """
        Center according to the mean image calculated using the training data
        Alexnet style
        :param X: numpy array same number of features as the training data
        :return: Centered dataset
        """

        if X:
            return X - self.meanImageAlex
        else:
            self.X = self.X - self.meanImageAlex
            self.X_val = self.X_val - self.meanImageAlex
            print 'Training data has been centered alexnet style'

    def center_VGG(self, X=None):
        """
        Center according to the mean image calculated using the training data
        VGG style
        :param X: numpy array same number of features as the training data
        :return: Centered dataset
        """
        if X:
            return X - self.meanImageVGG
        else:
            self.X = self.X - self.meanImageVGG
            self.X_val = self.X_val - self.meanImageVGG
            print 'Training data has been centered VGG style'

    def sobel_image(self):
        # replace images with the output of sobel filter on each image
        self.df['Image'] = self.df['Image'].apply(lambda im: sobel(im.reshape(96, 96)).reshape(-1))
        self._extract_Xy(col='Image')
        
    def hist_eqal_image(self):
        """"Extract histogram equalized images"""
        self.df['Image'] = self.df['Image'].apply(lambda im: equalize_hist(im.reshape(96, 96)).reshape(-1))
        self._extract_Xy(col='Image')

    def stack_origi_sobel(self):
        """stack original image with """
        df_preproc = pd.DataFrame(self.df['Image'])
        df_preproc['sobelh'] = df_preproc['Image'].apply(lambda im: sobel_h(im.reshape(96, 96)).reshape(-1))
        df_preproc['sobelv'] = df_preproc['Image'].apply(lambda im: sobel_v(im.reshape(96, 96)).reshape(-1))
        col = 'Image'
        self.X = np.vstack(df_preproc[col].values).reshape(-1,1,96,96)
        self.y = self.df[self.df.columns[:-1]].values
        col = 'sobelh'
        tempx1 = np.vstack(df_preproc[col].values).reshape(-1, 1, 96, 96)
        col = 'sobelv'
        tempx2 = np.vstack(df_preproc[col].values).reshape(-1, 1, 96, 96)
        self.X = np.concatenate((self.X,tempx1,tempx2), axis=1)


def reinitiate_set_params(network,
                          weights = None):
        # change weights of a trained network to a random set or a user defined value
        # useful in case of big networks and cross validation
        # instead of the long time of recompiling you can just
        # re-init the network weights
        if not weights:
            old = get_all_param_values(network)
            weights = []
            for layer in old:
                shape = layer.shape
                if len(shape)<2:
                    shape = (shape[0], 1)
                W= GlorotUniform()(shape)
                if W.shape != layer.shape:
                    W = np.squeeze(W, axis= 1)
                weights.append(W)
        set_all_param_values(network, weights)
        return network


def shared_dataset(X,y,borrow=True):
    """

    :param X: array like to be shared in theano
    :param y: array like to be shared in theano
    :param borrow: borrow option of theano.shared
    :return: shared version of X,y
    """
    shared_x=theano.shared(np.asarray(X,
                                      dtype=theano.config.floatX),
                           borrow=borrow)
    shared_y= theano.shared(np.asarray(y,
                                       dtype=theano.config.floatX),
                            borrow=borrow)
    return shared_x, shared_y


def build_update_functions(train_set_x, train_set_y,
                           valid_set_x, valid_set_y,
                           network,
                           y, X,
                           train_MASK, val_MASK,
                           batch_size=32,
                           l2_reg=.0001,
                           learning_rate=.005,
                           momentum=.9):
    # build update functions
    # extract tensor representing the network predictions
    prediction = get_output(network)
    ################################################
    ##################old###########################
    # # collect squared error
    # loss_RMSE = squared_error(prediction, y)
    # # compute the root mean squared error
    # loss_RMSE = loss_RMSE.mean().sqrt()
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
    l2_penalty = regularize_network_params(network, l2)
    loss = (1 - l2_reg) * loss_RMSE + l2_reg * l2_penalty
    # get network params
    params = get_all_params(network)

    #     # create update criterion
    # print('nestrov')
    # updates = nesterov_momentum( loss, params, learning_rate=.01, momentum=.9)

    # print('AdaGrad')
    # updates = adagrad(loss, params,learning_rate= 1e-2)
    #
    print('RMSPROP \n')
    updates = rmsprop(loss, params, learning_rate=1e-3)
    # create validation/test loss expression
    # the loss represents the loss for all the labels
    test_prediction = get_output(network, deterministic=True)
    ################################################
    ##################old###########################
    #     # collect squared error
    #     test_loss = squared_error(test_prediction,y)
    #     # compute the root mean squared error
    #     test_loss = test_loss.mean().sqrt()
    # #     test_loss_withl2 = (1-l2_reg) * test_loss + l2_reg * l2_penalty
    ################################################
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


def early_stop_train(train_set_x,train_set_y,
                     valid_set_x,valid_set_y,
                     network,train_fn,val_fn,batch_size = 32):
    """Get the network and update functions as input and apply early stop training.
    Should return a trained network with training history.
    ----------------------
    Input
    ----------------------
    :train_set_x: Training samples, loaded to GPU by theano.shared()
    :valid_set_x: Test samples, loaded to GPU by theano.shared()
    :train_set_y: Training outputs, loaded to GPU by theano.shared()
    :valid_set_y: Training outputs, loaded to GPU by theano.shared()
    :network: Deep model, the output layer of the network build using lasagne
    :train_fn: theano.function to update the network
    :val_fn: theano.function to calculate validation loss
    ----------------------
    Outputs
    ----------------------
    train_loss_history
    val_loss_history_
    network
    ----------------------
    """
    # network parameters
    # TODO: for testing hyper parameters, n_iter set to 400
    n_iter = 2000
    improvement_threshold = 0.998
    patience = 40000
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size + 1
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size + 1
    patience_increase = .3
    validation_frequency = min(n_train_batches, patience // 10)
    print 'validation_frequency',validation_frequency
    train_loss_history_temp = []
    best_val_loss_ = np.inf
    epoch = 0
    done_looping = False
    train_loss_history_ = []
    val_loss_history_ = []

    print 'start training'
    print 'shape training', train_set_x.get_value(borrow=True).shape, '\n'
    print 'shape validation', valid_set_x.get_value(borrow=True).shape, '\n'
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


                    # save the best model as pickle file
                    pickle.dump([best_network_params,best_val_loss_,best_epoch_,
                                 train_loss_history_,val_loss_history_,network],
                                open("results.p", "wb"))

            # check if patience exceeded and set the training loop to stop
            if (patience <= num_minibatch_checked):
                print 'patience reached \n'
                # reset the network weights to the best params saved
                print 'resetting the network params to that of the best seen \n'
                reinitiate_set_params(network=network,
                                      weights=best_network_params)
                # done optimising, break the optimisation loop
                done_looping = True
                break

        freq = 1
        if (epoch % freq) == 0:
            print (('epoch %i, validation loss %f, training loss %f, patience %i, in %f secs \n') %
                   (epoch, current_val_loss,train_loss_history_[-1], patience, time.time() - start_time))
            start_time = time.time()

    return 0
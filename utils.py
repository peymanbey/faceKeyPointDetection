from os.path import expanduser
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import theano
from lasagne.layers import  set_all_param_values

class data_set(object):
    """
    Store path to train and test date. Provide preprocessed data
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
        print '{} samples from the total of {} have missing values'.format(
            self.df.isnull().any(axis=1).sum(), self.df.shape[0])
        print 'Missing values appear in {} different columns of output targets'.format(
            self.df.isnull().any(axis=0).sum())
        # extract X,y
        self._extract_Xy()
        print 'shape of X', self.X.shape, 'and y', self.y.shape

    def _extract_Xy(self):
        # extract X and y
        self.X = np.vstack(self.df['Image'].values).astype(np.float32)
        self.y = self.df[self.df.columns[:-1]].values.astype(np.float32)
        # Shuffle the data
        self.X, self.y = shuffle(self.X, self.y, random_state=47)
        # Calculate mean image
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
        Center X according to the mean image calculated using the training data
        Input: X, numpy array same number of features as the training data
        output: X centered according to alexnet mean image
        """
        if X:
            return X - self.meanImageAlex
        else:
            self._extract_Xy()
            self.X = self.X - self.meanImageAlex
            print 'Training data has been centered alexnet style'

    def center_VGG(self, X=None):
        """
        Center X according to the mean value calculated using the training data
        Input: X, numpy array same number of features as the training data
        output: X centered according to VGG mean value
        """
        if X:
            return X - self.meanImageVGG
        else:
            self._extract_Xy()
            self.X = self.X - self.meanImageVGG
            print 'Training data has been centered VGG style'


def reinitiate_set_params(network,
                          weights = None):
        # change weights of a trained network to a random set or a user defined value
        # useful in case of big networks and cross validation
        # instead of the long time of recompiling you can just
        # re-init the network weights
        if not weights:
            old = lasagne.layers.get_all_param_values(network)
            weights = []
            for layer in old:
                shape = layer.shape
                if len(shape)<2:
                    shape = (shape[0], 1)
                W= lasagne.init.GlorotUniform()(shape)
                if W.shape != layer.shape:
                    W = np.squeeze(W, axis= 1)
                weights.append(W)
        set_all_param_values(network, weights)
        return network

def shared_dataset(X,y,borrow=True):
    """Load data into shared variables
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
                           batch_size=32,
                           l2_reg=.01,
                           learning_rate=.005,
                           momentum=.9):


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
    l2_penalty = regularize_network_params(network, l2)
    loss = (1 - l2_reg) * loss_RMSE + l2_reg * l2_penalty
    # get network params
    params = get_all_params(network)

    #     # create update criterion
    #     print('nestrov')
    #     updates = nesterov_momentum(
    #         loss, params, learning_rate=learning_rate, momentum=momentum)

    #     print('AdaGrad')
    #     updates = adagrad(loss, params,learning_rate= 1e-3)

    print('RMSPROP')
    updates = rmsprop(loss, params, learning_rate=1e-3)
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
                               outputs=[loss, loss_RMSE],
                               updates=updates,
                               givens={X: train_set_x[
                                          index * batch_size: T.minimum((index + 1) * batch_size, train_set_x_size)],
                                       y: train_set_y[
                                          index * batch_size: T.minimum((index + 1) * batch_size, train_set_x_size)]})
    # validation function
    val_fn = theano.function(inputs=[index],
                             outputs=[test_loss, prediction],
                             givens={
                                 X: valid_set_x[index * batch_size: T.minimum((index + 1) * batch_size, val_set_x_size)],
                                 y: valid_set_y[index * batch_size: T.minimum((index + 1) * batch_size, val_set_x_size)]})
    return train_fn, val_fn
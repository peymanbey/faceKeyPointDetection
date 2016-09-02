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
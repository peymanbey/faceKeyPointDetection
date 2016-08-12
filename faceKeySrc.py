import os
import numpy
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
import matplotlib.pyplot as plt

# general load and preparing data

def load_data(path, test=False, col=None):
    """ Load the data from path
        by default it assums the training data and
        loads all the columns
    """
    df = read_csv(os.path.expanduser(path))
    # the Image column is the pixel values separated by space
    # convert the values to numpy array
    df['Image'] = df['Image'].apply(lambda im: numpy.fromstring(im, sep=' '))
    
    # if you want only a subset of columns, passed as col to input
    if col:
        df = df[list(col)+['Image']]
    
    # some key-points have missing values
    # deal with them in handle_missing
    # print(df.count())
    df = handle_missing_values(df)
    # print(df.count())
    
    # the Image column contains pixel values 
    # it is a list separated by space
    # convert it into numpy array using np.vstack
    # also scale them to [0, 1]
    X = numpy.vstack(df['Image'].values) / 255.
    
    # convert values to float32
    X = X.astype(numpy.float32)
    
    # for training data, manipulate target values
    # scale the target values
    # shuffle data
    # Convert it to float 32
    if not test:# only train data has y columns
        Y = df[df.columns[:-1]].values
        Y = (Y - 48) / 48  # scale target coordinates to [-1, 1]
        X, Y = shuffle(X, Y, random_state =54)  # shuffle train data
        Y = Y.astype(numpy.float32)
    else:
        Y = None

    return X, Y

def handle_missing_values(df):
    """For the time being, just drop all the samples with missing values
    """
    print("just drop all the samples with missing values, consider a better approach")
    newdf= df.dropna()
    return newdf

def plot_train_valid(history_train_loss, history_validation_loss):
    plt.figure(figsize=(8, 8))
    valid_train_curves=plt.subplot(111)
    valid_train_curves.plot(history_validation_loss, label="validation loss")
    valid_train_curves.plot(history_train_loss, label="training loss")
    valid_train_curves.legend(loc='best')
    valid_train_curves.set_xlabel('epoch')
    valid_train_curves.set_ylabel('MSE loss')
    valid_train_curves.set_title('non-regularized linear regression')

def shared_dataset(X,y,borrow=True):
    """Load data into shared variables    
    """
    shared_x=theano.shared(numpy.asarray(X,
                                        dtype=theano.config.floatX),
                          borrow=borrow)
    shared_y= theano.shared(numpy.asarray(y,
                                        dtype=theano.config.floatX),
                          borrow=borrow)
    return shared_x, shared_y

# reusable classes and functions

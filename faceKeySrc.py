import os
import numpy
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
import theano.tensor as T


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
    
    # some keypoints have missing valuses
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
    if not test:
        Y = df[df.columns[:-1]].values
        Y = (Y - 48) / 48  # scale target coordinates to [-1, 1]
        X, Y = shuffle(X, Y, random_state =54)  # shuffle train data
        Y = Y.astype(numpy.float32)
    else:
        Y = None

    return X, Y

def plot_train_valid(history_train_loss, history_validation_loss):
    plt.figure(figsize=(8, 8))
    valid_train_curves=plt.subplot(111)
    valid_train_curves.plot(history_validation_loss, label="validation loss")
    valid_train_curves.plot(history_train_loss, label="training loss")
    valid_train_curves.legend(loc='best')
    valid_train_curves.set_xlabel('epoch')
    valid_train_curves.set_ylabel('MSE loss')
    valid_train_curves.set_title('non-regularized linear regression')

def handle_missing_values(df):
    """For the time being, just drop all the samples with missing values
    """
    newdf= df.dropna()
    return newdf



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


import matplotlib.pyplot as plt

class linear_regresion(object):
    """multi-target linear regression 
    Fully described with weight matrix :math:'W'
    and bias vectir :math:'b'.       
    """
    def __init__(self, input, n_in,n_out):
        """initialize parameters of linear regression
        :type input: theano.tensir.TensorType
        :param input: the symbolic variable that describes
        the input of the architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimesion of
        the space data points lie in
        
        :type n_out: int 
        :param n_out: number of output units, the number of
        target variables to predict
        
        """
        
        # initializing the weghts matrix by zero and shape(n_in,n_out)
        self.W= theano.shared(
            value=numpy.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize bias
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
       
        # symbolic expression of computing the output using W and b
        self.y_pred=T.dot(input,self.W)+self.b# make sure it is correct
        
        # parameters of the model
        self.param=[self.W,self.b]
        
        # keep track of model input
        self.input=input
        
        # define the loss function
    def loss_MSE(self,y):
        """returns the MSE error of prediction of the model
        :type y: theano.tensor.TensorType
        :param y: the vector that gives each samples correct prediction value
        """
        #  T.sum(T.sqr(targets-outputs),axis=1) 
        # I use averaging to         
        return T.mean(T.sqr(y-self.y_pred))#,axis=[0,1])
    def errors(self, y):
        """return the number of errors in minibatch
        
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example 
        the correct target values
        """
        # check if the dimension of y and y_pred is the same
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y',y.type, 'y_pred', self.y_pred.type)
            )
        return T.mean(T.neq(self.y_pred,y))
        
def train_early_stopping(train_model,
                         validate_model,
                         n_epochs,
                         n_train_batches,
                         n_valid_batches,
                         patience = 1000,  # look as this many examples regardless,
                         improvement_threshold = 0.995,  # a relative improvement of this much is
                                                        # considered significant
                         patience_increase = 2  # wait this much longer when a new best is
                        ):
    
    validation_frequency = min(n_train_batches, patience // 10)
                                # go through this many
                                # minibatche before checking the network
                                # on the validation set; in this case we
                                # check every epoch
    
    best_validation_loss = numpy.inf
    test_score = 0.
    # start_time = timeit.default_timer()
    
    done_looping = False
    epoch = 0
    history_validation_loss=numpy.inf#numpy.empty([])
    history_train_loss=numpy.inf#numpy.empty([])
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost =  train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                history_validation_loss = numpy.append(history_validation_loss,numpy.mean(validation_losses))
                
                train_losses = [train_model(i)
                                for i in range(n_train_batches)]
                history_train_loss = numpy.append(history_train_loss,numpy.mean(train_losses))
                #             print(
                #                 'epoch %i, minibatch %i/%i, validation error %f %%' %
                #                 (
                #                     epoch,
                #                     minibatch_index + 1,
                #                     n_train_batches,
                #                     this_validation_loss 
                #                 )
                #             )
                if patience <= iter:
                    #one_looping = True
                    break
    return history_train_loss, history_validation_loss
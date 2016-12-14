"""
Generate the submission csv file
"""
import cPickle as pickle
from os.path import  expanduser
from utils import data_set,  reinitiate_set_params
import numpy as np
import pandas as pd
import theano
from lasagne.layers import  get_output, get_all_layers, get_all_param_values, NonlinearityLayer
from skimage.filters import sobel, sobel_h, sobel_v


def build_test_func(test_set_x,
                    network,X):
    prediction = get_output(network, deterministic=True)
    test_fc = theano.function(inputs=[X],
                              outputs=prediction)
    print "Compiling the test function is done \n"
    return test_fc

def load_model_predict(PATH_simresult, test_set_X):
    # Load sim results
    print 'loading', PATH_simresult, '\n'
    with open(PATH_simresult, "rb") as f:
        temp = pickle.load(f)
        network = temp[-1]
        
    best_network_params = get_all_param_values(network)
    # extract input var
    print 'extract input var \n'
    X = get_all_layers(network)[0].input_var

    # build test function
    print 'build test function and reinit network \n'
    test_fn = build_test_func(test_set_X,
                              network, X)

    reinitiate_set_params(network,
                          weights=best_network_params)

    print 'test set shape', test_set_X.shape, 'type:', type(test_set_X), '\n'

    print 'make prediction \n'
    # predictedy = test_fn(test_set_X)
    # batched implementation
    batch_size = 128
    n_test_batches = test_set_X.shape[0] // batch_size + 1
    test_set_x_size = test_set_X.shape[0]
    predictedy = [test_fn(
        test_set_X[index * batch_size: min((index + 1) * batch_size, test_set_x_size)])
                  for index in range(n_test_batches)]

    predictedy = np.vstack(predictedy)
    return  predictedy


def stack_origi_sobel(df):
    """stack original image with """
    df_preproc = pd.DataFrame(df['Image'])
    df_preproc['sobelh'] = df_preproc['Image'].apply(lambda im: sobel_h(im.reshape(96, 96)).reshape(-1))
    df_preproc['sobelv'] = df_preproc['Image'].apply(lambda im: sobel_v(im.reshape(96, 96)).reshape(-1))
    col = 'Image'
    X = np.vstack(df_preproc[col].values).reshape(-1, 1, 96, 96)
    col = 'sobelh'
    tempx1 = np.vstack(df_preproc[col].values).reshape(-1, 1, 96, 96)
    col = 'sobelv'
    tempx2 = np.vstack(df_preproc[col].values).reshape(-1, 1, 96, 96)
    X = np.concatenate((X, tempx1, tempx2), axis=1).astype(np.float32)
    return X

if __name__ == "__main__":

    # prepare submission tables
    PATH_train = "../data/training.csv"
    PATH_test = "../data/test.csv"

    # Get sequence of column names from training data
    df = pd.read_csv(expanduser(PATH_train))

    # convert feat name to dict for fast access
    featNameid = {}
    for c, key in enumerate(list(df.columns[:-1])):
        featNameid[key] = c

    # read the submission file guide
    df = pd.read_csv(expanduser("../data/IdLookupTable.csv"))

    # replace the feature names with their output index
    temp = df['FeatureName'].values
    for i, val in enumerate(temp):
        temp[i] = featNameid[val]

    # load train and test data
#    print 'load test train data \n'
#    data = data_set(path_train=PATH_train, path_test=PATH_test)
    # data.split_trainval()
#    testdf = pd.read_csv(expanduser(PATH_test))
    # normal image
#    testdf['Image']  = testdf['Image'].apply(lambda im: np.fromstring(im, sep=' '))
#    imID = testdf['ImageId'].values


#    test_set_X = np.vstack(testdf['Image'].values).astype(np.float32)
    # apply pre-processing if needed
#    test_set_X = test_set_X.reshape(-1,1,96,96)
    # print 'alex pre-proc \n'
    # test_set_X = data.center_alexnet(test_set_X)
#    print 'VGG pre-proc \n'
#    test_set_X = test_set_X - data.X.mean()

#    PATH_simresult = "simResults/results10.p"
#    predictedy = load_model_predict(PATH_simresult, test_set_X)
#    PATH_simresult = "simResults/results6.p"
#    predictedy += load_model_predict(PATH_simresult, test_set_X)
#    PATH_simresult = "simResults/results7.p"
#    predictedy += load_model_predict(PATH_simresult, test_set_X)
#    PATH_simresult = "simResults/results9.p"
#    predictedy += load_model_predict(PATH_simresult, test_set_X)
#    # predictedy /= 4

    ###########################################################################################
#    # sobel image
#    print 'load test train data \n'
#    data = data_set(path_train=PATH_train, path_test=PATH_test)
#    data.sobel_image()
#
#    testdf = pd.read_csv(expanduser(PATH_test))
#    testdf['Image'] = testdf['Image'].apply(lambda im: np.fromstring(im, sep=' '))
#    imID = testdf['ImageId'].values
#    testdf['Image'] = testdf['Image'].apply(lambda im: sobel(im.reshape(96, 96)).reshape(-1))
#    test_set_X = np.vstack(testdf['Image'].values).astype(np.float32)
#    test_set_X = test_set_X.reshape(-1, 1, 96,96)
#
#    PATH_simresult = "simResults/results11.p"
#    predictedy += load_model_predict(PATH_simresult, test_set_X)
#    # predictedy /= 5
    ##########################################################################################

    ##########################################################################################
    # EXPERIMENT 12
    # sobel stacking
    # it does have reshaping inside no need to latter reshape step
    testdf = pd.read_csv(expanduser(PATH_test))
    testdf['Image'] = testdf['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    imID = testdf['ImageId'].values
    test_set_X = stack_origi_sobel(testdf)

    # PATH_simresult = "simResults/results12.p"
    # predictedy = load_model_predict(PATH_simresult, test_set_X)
    PATH_simresult = "simResults/results15.p"
    predictedy = load_model_predict(PATH_simresult, test_set_X)
    PATH_simresult = "simResults/results14.p"
    predictedy += load_model_predict(PATH_simresult, test_set_X)
    PATH_simresult = "simResults/results13.p"
    predictedy += load_model_predict(PATH_simresult, test_set_X)
    predictedy /= 3
    # ###########################################################################################

    print 'shape of prediction', predictedy.shape,\
        'max:', predictedy.max(), 'min:', predictedy.min(),'\n'

    # insert predictions into submissioin table
    print df[['RowId', 'Location']][:10]
    print 'fill submission table'

    temp = df.values
    for i in xrange(temp.shape[0]):
        temp[i, 3] = predictedy[temp[i, 1] - 1, temp[i, 2]]
    df['Location'] = temp[:, 3]

    print df[['RowId', 'Location']][:10]

    print 'write to submission file'
    df[['RowId', 'Location']].to_csv('Submission.csv', index=False)

    print df.loc[df['Location']>96]



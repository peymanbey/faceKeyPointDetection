# %%
# Import libraries that you want to use
import pandas as pd
import numpy as np
from os.path import expanduser


""" Kaggle Face Key Points data for learning CNNs.
 This code is based on various tutorials to design and train a CNN for facilal
 key point recognition.
 """

"""
 All tasks:
     1. Read the training data and transform it to numpy
     2. Do pre-processing and generate multiple versions of data:
          a. Exclude samples with missing values
          b. Normalize the data
          c. Image whitening for de-correlating pixels
          d. Missing value imputation
          e. Multi-label learning handling of missing values
          f. Augment the training images by
                i. Flipping around the vertical axis
                ii. Shifting up, down, left, right?
                iii. What else? Search for it
          g. Cluster instances, in terms of their annotation type.
          Apparently two different protocol has been used for annotation.
          Maybe using a different estimator for each would help to improve.
     3. Choose a CNN model, something that is available in pretrained format
     and use it as feature selector. then:
            a. Apply linear model on the extracted features
            b. Apply random forest on the extracted features
            c. Apply XGBoost on extracted features
            d. Fine tune the model to your data
                i. Use RELU
                ii. Use leaky-RELU with trainable parameter
     4. Write the script to predict and generate the submission file
     5. Track the leader board score for each combination of pre-processing
     and estimator
 """


def load_data(PATH):
    """"""
    df = pd.read_csv(expanduser(PATH))
    return 1

if __name__ == "__main__":
    # path to train and testing data
    PATH_train = "../data/training.csv"
    PATH_test = "../data/test.csv"
    # load training data
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import pandas as pd 

#X = pd.read_csv('test_sample.csv')
#X.drop('Unnamed: 0', axis=1, inplace=True)

def predict_location(X, model='Model_01.h5'):
    
    # clean dataset
    for i in range(0,2048):
        X[str(i)] = pd.to_numeric(X[str(i)], errors='coerce')
    X.dropna(axis=1, inplace=True)

    if len(X) < 1:
        print('Error')
        return 'Error'

    X.drop('ID', axis=1, inplace=True)

    # load the model from disk
    loaded_model = load_model(model)
    #loaded_model.load_weights('NN_Models/test.hdf5')

    # Generate predictions
    predictions = loaded_model.predict(X)
    print('Predictions: ', predictions, sep='\n', end='\n')

    return predictions

#predictions = predict_location(X)

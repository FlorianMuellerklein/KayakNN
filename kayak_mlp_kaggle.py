import kayak
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from sklearn.preprocessing import LabelBinarizer

batch_size     = 256
learn_rate     = 0.005
momentum       = 0.9
layer1_size    = 2000
layer2_size    = 1000
layer1_dropout = 0.25
layer2_dropout = 0.25
iterations     = 100

IMAGE_WIDTH = 28

binary = LabelBinarizer()

def nudge_dataset(X, Y):
    nudge_size = 1
    direction_matricies = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    scaled_direction_matricies = [[[comp*nudge_size for comp in vect] for vect in matrix] for matrix in direction_matricies]
    shift = lambda x, w: convolve(x.reshape((IMAGE_WIDTH, IMAGE_WIDTH)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in scaled_direction_matricies])

    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

def load_train_data():
    data = np.genfromtxt('Data/train.csv', delimiter = ',', skip_header = 1)

    # first ten values are the one hot encoded y (target) values
    y = data[:,0]
    binary.fit(y)
    y = binary.transform(y)

    data = data[:,1:] # x data
    data /= 255.0 # scale the data so values are between 0 and 1

    return data, y
    
def load_test_data():
    data = np.genfromtxt('Data/test.csv', delimiter = ',', skip_header = 1)
    
    data /= 255.0
    
    return data
    
def kayak_mlp(X, y):
    """
    Kayak implementation of a mlp with relu hidden layers and dropout
    """
    # Create a batcher object.
    batcher = kayak.Batcher(batch_size, X.shape[0])
    
    # count number of rows and columns
    num_examples, num_features = np.shape(X)
    
    X = kayak.Inputs(X, batcher)
    T = kayak.Targets(y, batcher)
    
    # ----------------------------- first hidden layer -------------------------------
    
    # set up weights for our input layer
    # use the same scheme as our numpy mlp
    input_range = 1.0 / num_features ** (1/2)
    weights_1 = kayak.Parameter(0.1 * np.random.randn (X.shape[1], layer1_size))
    bias_1 = kayak.Parameter(0.1 * np.random.randn(1, layer1_size))
    
    # linear combination of weights and inputs
    hidden_1_input = kayak.ElemAdd(kayak.MatMult(X, weights_1), bias_1)
    
    # apply activation function to hidden layer
    hidden_1_activation = kayak.HardReLU(hidden_1_input)
    
    # apply a dropout for regularization
    hidden_1_out = kayak.Dropout(hidden_1_activation, layer1_dropout, batcher = batcher)
    
    
    # ----------------------------- second hidden layer -----------------------------
    
    # set up weights
    weights_2 = kayak.Parameter(0.1 * np.random.randn(layer1_size, layer2_size))
    bias_2 = kayak.Parameter(0.1 * np.random.randn(1, layer2_size))
    
    # linear combination of weights and layer1 output
    hidden_2_input = kayak.ElemAdd(kayak.MatMult(hidden_1_out, weights_2), bias_2)
    
    # apply activation function to hidden layer
    hidden_2_activation = kayak.HardReLU(hidden_2_input)
    
    # apply a dropout for regularization
    hidden_2_out = kayak.Dropout(hidden_2_activation, layer2_dropout, batcher = batcher)
    
    # ----------------------------- output layer -----------------------------------
    
    weights_out = kayak.Parameter(0.1 * np.random.randn(layer2_size, 10))
    bias_out = kayak.Parameter(0.1 * np.random.randn(1, 10))
    
    # linear combination of layer2 output and output weights
    out = kayak.ElemAdd(kayak.MatMult(hidden_2_out, weights_out), bias_out)
    
    # apply activation function to output
    yhat = kayak.LogSoftMax(out)
    
    # ----------------------------- loss function -----------------------------------
    
    loss = kayak.MatSum(kayak.LogMultinomialLoss(yhat, T))

    # Use momentum for the gradient-based optimization.
    mom_grad_W1 = np.zeros(weights_1.shape)
    mom_grad_W2 = np.zeros(weights_2.shape)
    mom_grad_W3 = np.zeros(weights_out.shape)

    # Loop over epochs.
    for epoch in xrange(iterations):

        # Track the total loss.
        total_loss = 0.0
        
        for batch in batcher:
            # Compute the loss of this minibatch by asking the Kayak
            # object for its value and giving it reset=True.
            total_loss += loss.value

            # Now ask the loss for its gradient in terms of the
            # weights and the biases -- the two things we're trying to
            # learn here.
            grad_W1 = loss.grad(weights_1)
            grad_B1 = loss.grad(bias_1)
            grad_W2 = loss.grad(weights_2)
            grad_B2 = loss.grad(bias_2)
            grad_W3 = loss.grad(weights_out)
            grad_B3 = loss.grad(bias_out)
        
            # Use momentum on the weight gradients.
            mom_grad_W1 = momentum * mom_grad_W1 + (1.0 - momentum) * grad_W1
            mom_grad_W2 = momentum * mom_grad_W2 + (1.0 - momentum) * grad_W2
            mom_grad_W3 = momentum * mom_grad_W3 + (1.0 - momentum) * grad_W3

            # Now make the actual parameter updates.
            weights_1.value   -= learn_rate * mom_grad_W1
            bias_1.value      -= learn_rate * grad_B1
            weights_2.value   -= learn_rate * mom_grad_W2
            bias_2.value      -= learn_rate * grad_B2
            weights_out.value -= learn_rate * mom_grad_W3
            bias_out.value    -= learn_rate * grad_B3

        print epoch, total_loss
        
    def compute_predictions(x):
        X.data = x
        batcher.test_mode()
        return yhat.value

    return compute_predictions
        
       
def main():
    # set up parameters for the neural network, these variables are called in the kayak_mlp function
    
    X_train, y_train = load_train_data()
    X_train, y_train = nudge_dataset(X_train, y_train)
    
    pred_func = kayak_mlp(X_train, y_train)
    
    # Make predictions on the test data.
    X_test = load_test_data()
    preds = np.array(pred_func( X_test ))
    preds = binary.inverse_transform(preds)
    preds = preds.astype(int)

    # How did we do?
    print preds
    
    # load benchmark csv to use as output dataframe
    out = pd.read_csv('Data/knn_benchmark.csv')
    out['Label'] = preds
    
    # save predictions as csv for kaggle
    out.to_csv('kayak_deep_preds.csv', index = False)
    
if __name__ == '__main__':
    main()
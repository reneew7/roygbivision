#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Collaborated with Anne Pham
// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                m.data[i][j] = 1 / (1 + exp(-x));
            } else if (a == RELU){
                m.data[i][j] = MAX(x, 0);
            } else if (a == LRELU){
                m.data[i][j] = MAX(x, 0.1 * x);
            } else if (a == SOFTMAX){
                m.data[i][j] = exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // have to normalize by sum if we are using SOFTMAX

            for (j = 0; j < m.cols; ++j) {
                m.data[i][j] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // multiply the correct element of d by the gradient
            if (a == LOGISTIC) {
                d.data[i][j] *= x * (1 - x);
            } else if (a == RELU) {
                if (x > 0) {
                    d.data[i][j] *= 1;
                } else {
                    d.data[i][j] = 0;
                }
            } else if (a == LRELU) {
                if (x > 0) {
                    d.data[i][j] *= 1;
                } else {
                    d.data[i][j] *= 0.1;
                }
            }
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation


    // multiply input by weights and apply activation function.
    matrix out = matrix_mult_matrix(l->in, l->w);
    activate_matrix(out, l->activation);


    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // modify it in place to be dL/d(xw)
    gradient_matrix(l->out, l->activation, delta);


    // 1.4.2
    // then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    matrix xt = transpose_matrix(l->in);
    matrix dw = matrix_mult_matrix(xt, delta); 
    l->dw = dw;

    
    // 1.4.3
    // finally, calculate dL/dx and return it.
    matrix wt = transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta, wt); // replace this
    free_matrix(wt);

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    matrix inner = axpy_matrix(-decay, l->w, l->dw);
    matrix dwt = axpy_matrix(momentum, l->v, inner);
    free_matrix(l->v);
    // save it to l->v
    l->v = dwt;

    // Update l->w
    matrix w_t1 = axpy_matrix(rate, l->v, l->w);
    free_matrix(l->w);
    l->w = w_t1;


    // Remember to free any intermediate results to avoid memory leaks

}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions 
//
// 5.2.2.1 Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
// Training accuracy tells us how well our model does on the training data, while testing accuracy tells us how well our model does on new data.
// Both are important metrics, but our testing accuracy is especially important to note because we don't want our model to overfit to the training
// data.
//
// 5.2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// Learning Rates and Training/Test Accuracy:
// learning rate 0.001
// training accuracy: %f 0.8579333333333333
// test accuracy:     %f 0.8674
// learning rate 0.01
// training accuracy: %f 0.9040166666666667
// test accuracy:     %f 0.9098
// learning rate 0.1
// training accuracy: %f 0.9216833333333333
// test accuracy:     %f 0.9189
// learning rate 1
// training accuracy: %f 0.8788166666666667
// test accuracy:     %f 0.8771
// learning rate 10
// training accuracy: %f 0.09871666666666666
// test accuracy:     %f 0.098
// Based on these results, it looks like training and test accuracy improve and worsen together based on learning rate. They look to be really close to each
// other in values, regardless of the learning rate. Accuracy in general seems to increase as learning rate increases up to a certain point, specifically
// when learning rate is 0.1. From 0.001 to 0.1, both accuracies grow steadily until they are best at 0.1. After that, when learning rate increased more to 1
// and 10, accuracy worsened again, particular at learning rate 10, when the model did very poorly.
//
// 5.2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// decay 0.0
// training accuracy: %f 0.9040166666666667
// test accuracy:     %f 0.9098
// decay 0.1
// training accuracy: %f 0.9034
// test accuracy:     %f 0.912
// decay 0.01
// training accuracy: %f 0.9038333333333334
// test accuracy:     %f 0.9078
// decay 0.001
// training accuracy: %f 0.90285
// test accuracy:     %f 0.91
// decay 0.0001
// training accuracy: %f 0.90465
// test accuracy:     %f 0.9099
// decay 0.00001
// training accuracy: %f 0.9035666666666666
// test accuracy:     %f 0.9091
// It seems that from the data, decay has no real significant effect on training accuracy nor test accuracy. Across the different decay
// values, accuracy stayed roughly the same. The relationship between training and test accuracy was also pretty consistent across all
// decay values, as decay did not change the fact that these values are close together (at most off by 0.01).
//
// 5.2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// The RELU activation performed best.
// LOGISTIC
// training accuracy: %f 0.8881333333333333
// test accuracy:     %f 0.8957
// LINEAR
// training accuracy: %f 0.9148666666666667
// test accuracy:     %f 0.9172
// RELU
// training accuracy: %f 0.9233
// test accuracy:     %f 0.9264
// LRELU
// training accuracy: %f 0.9199
// test accuracy:     %f 0.92
// SOFTMAX
// training accuracy: %f 0.6528833333333334
// test accuracy:     %f 0.6615
//
// 5.2.3.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// The best learning rate with RELU activation was 0.1
// learning rate 0.001
// training accuracy: %f 0.8469666666666666
// test accuracy:     %f 0.8516
// learning rate 0.01
// training accuracy: %f 0.9233
// test accuracy:     %f 0.9264
// learning rate 0.1
// training accuracy: %f 0.9637333333333333
// test accuracy:     %f 0.9586
// learning rate 1
// training accuracy: %f 0.09871666666666666
// test accuracy:     %f 0.098
//
// 5.2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// Decay has no significant effect on the accuracy, so it does not help. One possible reason for this might be because we are only doing 1000
// iterations, so there are not enough iterations for decay to have a significant impact. We also only have 2 layers, so maybe adding more layers will
// cause decay to have a larger impact. A large decay like 1 does make accuracy decrease quite a bit, potentially just because it is too large.
// decay 0
// training accuracy: %f 0.9637333333333333
// test accuracy:     %f 0.9586
// decay 0.0001
// training accuracy: %f 0.9616333333333333
// test accuracy:     %f 0.956
// decay 0.001
// training accuracy: %f 0.9653333333333334
// test accuracy:     %f 0.9594
// decay 0.01
// training accuracy: %f 0.9593666666666667
// test accuracy:     %f 0.9553
// decay 0.1
// training accuracy: %f 0.9512666666666667
// test accuracy:     %f 0.9493
// decay 1
// training accuracy: %f 0.91585
// test accuracy:     %f 0.921
//
// 5.2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// decay 0.0001
// training accuracy: %f 0.9847833333333333
// test accuracy:     %f 0.9695
// decay 0.001
// training accuracy: %f 0.9830666666666666
// test accuracy:     %f 0.969
// decay 0.01
// training accuracy: %f 0.9803166666666666
// test accuracy:     %f 0.9675
// decay 0.1
// training accuracy: %f 0.9777
// test accuracy:     %f 0.9701
// decay 1
// training accuracy: %f 0.9283
// test accuracy:     %f 0.9268
// The model did best with decay 0.0001, with the highest training and test accuracy. It might have been best because it didn't need a lot of
// decay to do well, so the smallest amount actually worked best.
//
// 5.3.2.1 How well does your network perform on the CIFAR dataset?
// 0.47332 training and 0.4584 testing with 0.0001 decay and 0.01 learning rate
//

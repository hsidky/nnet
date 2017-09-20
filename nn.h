// Copyright (c) 2013, Manuel Blum
// All rights reserved.

#ifndef __NN_H__
#define __NN_H__

#define F_TYPE double

#include <Eigen/Core>
#include <vector>

typedef Eigen::Matrix<F_TYPE, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
typedef Eigen::Matrix<F_TYPE, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Array<F_TYPE, Eigen::Dynamic, Eigen::Dynamic> array_t;

struct nn_layer 
{
    size_t size;
    matrix_t a, z, delta;
    matrix_t W, dEdW, DeltaW, dEdb;
    vector_t b, Delab;
};

class neural_net 
{
protected:
    /** Allocate memory and initialize default values. */
    void init_layers(Eigen::VectorXi &topology);
    
    /** Holds the layers of the neural net. */
    std::vector<nn_layer> layers_;

    /** Holds the error gradient, jacobian, ... */ 
    matrix_t j_, jj_;
    vector_t je_;

    /** Number of adjustable parameters. */
    uint nparam_;

    /** Scaling parameters. */
    vector_t x_shift_;
    vector_t x_scale_;
    vector_t y_shift_;
    vector_t y_scale_;
    
public:
    /** Init neural net with given topology. */
    neural_net(Eigen::VectorXi& topology);
    
    /** Read neural net from file. */
    neural_net(const char* filename);

    /** Initial weights randomly (zero mean, standard deviation sd) . */
    void init_weights(F_TYPE sd);
    
    /** Propagate data through the net.
    *  Rows of X are instances, columns are features. */
    void forward_pass(const matrix_t& X);

    /** Compute NN loss w.r.t. input and output data.
      * Also backpropogates error. 
      */
    F_TYPE loss(const matrix_t& X, const matrix_t& Y);
    
    /** Return activation of output layer. */
    matrix_t get_activation();
    
    /** Get gradient of output(s) w.r.t. input i */
    matrix_t get_gradient(int index);
    
    /** Returns the logistic function values f(x) given x. */
    static matrix_t activation(const matrix_t& x);
    
    /** Returns the gradient f'(x) of the logistic function given f(x). */
    static matrix_t activation_gradient(const matrix_t& x);
    
    /** Compute autoscale parameters. */
    void autoscale(const matrix_t& X, const matrix_t& Y);
    
    /** Reset autoscale parameters */
    void autoscale_reset();      
    
    /** Write net parameter to file. */
    bool write(const char* filename);
    
    /** Destructor. */
    ~neural_net();
};

#endif

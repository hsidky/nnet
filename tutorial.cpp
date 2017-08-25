// Copyright (c) 2013, Manuel Blum
// All rights reserved.

#include <Eigen/Dense>
#include <iostream>
#include <cstdio>

#include "nn.h"

int main (int argc, const char* argv[]) 
{
    // input dimensionality
    int n_input = 2;
    // output dimensionality
    int n_output = 1;
    // number of training samples
    int m = 4;
    // number of layers
    int k = 3;
    
    // training inputs
    matrix_t X(m, n_input);
    matrix_t Y(m, n_output);
    
    // XOR problem
    X << 0, 0, 0, 1, 1, 0, 1, 1;
    Y << 0, 1, 1, 0;
    std::cout << "training input: " << std::endl << X << std::endl;
    std::cout << "training output: " << std::endl << Y << std::endl;
    
    // specify network topology
    Eigen::VectorXi topo(k);
    topo << n_input, 6, n_output;
    std::cout << "topology: " << std::endl << topo << std::endl;
    
    // initialize a neural network with given topology
    neural_net nn(topo);
    nn.autoscale(X,Y);
    
    // write model to disk
    nn.write("example.nn");
    
    // read model from disk
    neural_net nn2("example.nn");
    
    std::cout << "Loaded network" << std::endl;

    // testing 
    nn2.forward_pass(X);
    matrix_t Y_test = nn2.get_activation();
    
    std::cout << "test input:" << std::endl << X << std::endl;
    std::cout << "test output:" << std::endl << Y_test << std::endl;

    std::cout << "original network" << std::endl;
    
    // testing 
    nn.forward_pass(X);
    Y_test = nn.get_activation();
    
    std::cout << "test input:" << std::endl << X << std::endl;
    std::cout << "test output:" << std::endl << Y_test << std::endl;
    
    return 0;
}


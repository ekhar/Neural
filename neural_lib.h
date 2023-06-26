#define EULER_NUMBER 2.71828182846
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/*
-------------------------------
          STRUCTS
-------------------------------
*/
typedef struct {
  double activation;
  double z;
  double *weights;
  double bias;
  int num_weights;

  // derivatives
  double dactivation;
  double dz;
  double *dweights;
  double dbias;
} neuron;

typedef struct {
  bool output;
  double(*activation)(double);
  double(*dactivation)(double);
  int num_neurons;
  neuron *neurons;
} layer;

typedef struct {
  int num_layers;
  layer *layers;
} NN;

/*
-------------------------------
            MATH
-------------------------------
*/
double sigmoid(double x);
double dsigmoid(double x);
double relu(double x);
double drelu(double x);
double cost(double x, double y);
double dcost(double x, double y);

/*
-------------------------------
    INITIALIZATION/CLEANUP
-------------------------------
*/
NN Neural_Network(int num_layers, int *layers);
void init_weights(NN *z);
void free_NN(NN *z);
void set_inputs(NN *net, double *ins);

/*
-------------------------------
            LOGIC
-------------------------------
*/
void forward_prop(NN *n);
void backward_prop(NN *n, double *tv); 
void update_weights(NN *net, double alpha); 
void train_step(NN *z, double *inputs, double *outputs, double learning_rate);
void predict(NN *net, double *inputs);
double total_error(NN*net, double *tv);
/*
-------------------------------
            LOGGING
-------------------------------
*/
void printLayer(layer *l);
void printNN(NN *z);
void printdNN(NN *z);
void printOut(NN *n);
/*
-------------------------------
            TESTS
-------------------------------
*/
void test_init(NN *net);
void test_forward(NN *net, double *inputs);
void test_back(NN *net, double *tv);

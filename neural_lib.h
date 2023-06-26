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
  float activation;
  float z;
  float *weights;
  float bias;
  int num_weights;

  // derivatives
  float dactivation;
  float dz;
  float *dweights;
  float dbias;
} neuron;

typedef struct {
  bool output;
  float(*activation)(float);
  float(*dactivation)(float);
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
float sigmoid(float x);
float dsigmoid(float x);
float relu(float x);
float drelu(float x);
float cost(float x, float y);
float dcost(float x, float y);

/*
-------------------------------
    INITIALIZATION/CLEANUP
-------------------------------
*/
NN Neural_Network(int num_layers, int *layers);
void init_weights(NN *z);
void free_NN(NN *z);

/*
-------------------------------
            LOGIC
-------------------------------
*/
void forward_prop(NN *n);
void backward_prop(NN *n, float *tv); 
void update_weights(NN *net, float alpha); 
void train_step(NN *z, float *inputs, float *outputs, float learning_rate);
void predict(NN *net, float *inputs);
/*
-------------------------------
            LOGGING
-------------------------------
*/
void printLayer(layer *l);
void printNN(NN *z);
void printOut(NN *n);
/*
-------------------------------
            TESTS
-------------------------------
*/
void test_init(NN *net);
void test_forward(NN *net);
void test_back(NN *net);

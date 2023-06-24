#define EULER_NUMBER 2.71828182846
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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

  // temporary
  float *new_weights;
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
  void (*learning_alg)(layer *, layer *, float *);
  layer *layers;
} NN;

float sigmoid(float x);
float dsigmoid(float x);

float relu(float x);
float drelu(float x);

float cost(float x, float y);
float dcost(float x, float y);

void forward_prop(layer *x, layer *y);

// void cost(float x, float y);

// TODO
void backward_prop(NN *n, float *tv); 

void init_layer(layer *l);

// set up the NN

NN Neural_zwork(int num_layers, int *layers, const char* learning_alg);

void backward_prop(layer *lay_out, layer *layer_before, float *tv);

void init_weights(NN *z);

void train(NN *z, int *inputs, int *outputs, float learning_rate,
           char learning_alg);

void printLayer(layer *l);

void printNN(NN *z);

void free_NN(NN *z);

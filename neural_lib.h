#define EULER_NUMBER 2.71828182846
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct {
  float activation;
  float net;
  float *weights;
  float bias;
  int num_weights;

  // derivatives
  float dactivation;
  float dnet;
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

float relu(float x);

void forward_prop(layer *x, layer *y);

// void cost(float x, float y);

// TODO
void backward_prop(layer *lay_out, layer *layer_before, float *tv);

void init_layer(layer *l);

// set up the NN

NN Neural_Network(int num_layers, int *layers, const char* learning_alg);

void backward_prop(layer *lay_out, layer *layer_before, float *tv);

void init_weights(NN *net);

void train(NN *net, int *inputs, int *outputs, float learning_rate,
           char learning_alg);

void printLayer(layer *l);

void printNN(NN *net);

void free_NN(NN *net);

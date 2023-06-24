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
  layer *layers;
} NN;

float sigmoid(float x);
float dsigmoid(float x);

float relu(float x);
float drelu(float x);

float cost(float x, float y);
float dcost(float x, float y);

void forward_prop(NN *n);

// void cost(float x, float y);

void backward_prop(NN *n, float *tv); 

void init_layer(layer *l);

// set up the NN

NN Neural_Network(int num_layers, int *layers);

void init_weights(NN *z);

void train_step(NN *z, float *inputs, float *outputs, float learning_rate);

void printLayer(layer *l);

void printNN(NN *z);

void free_NN(NN *z);

void update_weights(NN *net, float alpha); 

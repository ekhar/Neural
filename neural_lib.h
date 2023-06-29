#ifndef NN_H
#define NN_H

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
  char * activation_name;
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
NN Neural_Network(int num_layers, int *layers, char hidden[], char output[]);
void init_weights(NN *net);
void free_NN(NN *net);
void set_inputs(NN *net, float *ins);
void set_layer_activation(layer *l, char activation_name[]);
/*
-------------------------------
            LOGIC
-------------------------------
*/
void forward_prop(NN *net);
void backward_prop(NN *net, float *tv); 
void update_weights(NN *net, float alpha); 
void train_step(NN *net, float *inputs, float *outputs, float learning_rate);
void predict(NN *net, float *inputs);
float total_error(NN*net, float *tv);
int max_output(NN* net);

/*
-------------------------------
            LOGGING
-------------------------------
*/
void printLayer(layer *l);
void printNN(NN *net);
void printdNN(NN *net);
void printOut(NN *net);


/*
-------------------------------
            Save State
-------------------------------
*/
void save_nn(NN *net, const char *filename);
void read_nn(NN *net, const char *filename);

#endif

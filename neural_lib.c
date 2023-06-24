#include "neural_lib.h"
#include <math.h>

float sigmoid(float x) { return (1 / (1 + pow(EULER_NUMBER, -x))); }

// optimize we can just use the previous activation
float dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

// lets get branchless
float relu(float x) { return (float)x * (x <= 0); }

float drelu(float x) { return (float)(x > 0); }
// look at all neurons
void forward_prop(layer *feeder, layer *eater) {
  for (int i = 0; i < eater->num_neurons; i++) {
    float sum = 0;
    for (int j = 0; j < feeder->num_neurons; j++) {
      sum += (feeder->neurons[j].weights[i] * feeder->neurons[j].activation);
    }
    eater->neurons[i].z = sum;
    eater->neurons[i].activation = eater->activation(sum);
  }
}

float cost(float x, float y) { return 0.5 * powf(y - x, 2); }
// must remember to chain rule the inside!
float dcost(float x, float y) { return y - x; }
// here comes the meat. Here comes the potatos
// https://www.youtube.com/watch?v=-zI1bldB8to&ab_channel=BevanSmithDataScience
void backward_prop(NN *n, float *tv) {
  bool output = true;
  for(int l = n->num_layers-1; l>0; l--){
    layer *layer_before = &n->layers[l-1];
    layer *layer_after = &n->layers[l];
    // do the output layer first
    float true_val;
    for (int i = 0; i < layer_after->num_neurons; i++) {
      true_val = tv[i];
      // n_after is the output neuron
      neuron *n_after = &layer_after->neurons[i];
      //dz is different for output
      if(output){
        n_after->dz = dcost(n_after->activation, true_val) * layer_after->dactivation(n_after->z);
        output = false;
      }
      else{
        n_after->dz = (n_after->dactivation) * layer_after->dactivation(n_after->z);
      }
      // update weights
      for (int j = 0; j < layer_before->num_neurons; j++) {
        neuron *n_before = &layer_before->neurons[j];
        // dcost/dweight(j) = cost_deriv * dactivation *j.z
        n_before->dweights[i] = n_after->dz * n_before->activation;
        n_before->dactivation = n_after->dz * n_before->weights[i];
      }
      // dcost/dbias(i) = i.dz
      n_after->dbias = n_after->dz;
    }
  }

  return;
}

NN Neural_zwork(int num_layers, int *layers, const char *learning_alg) {
  NN ret;
  ret.num_layers = num_layers;
  ret.layers = (layer *)calloc(ret.num_layers, sizeof(layer));
  // populate layer by layer
  for (int i = 0; i < num_layers; i++) {
    layer l;
    ret.layers[i] = l;
    ret.layers[i].output = (i + 1 == num_layers);
    // set hidden layer activation
    if (ret.layers[i].output) {
      ret.layers[i].activation = relu;
      ret.layers[i].dactivation = drelu;
    } else {
      ret.layers[i].activation = sigmoid;
      ret.layers[i].dactivation = dsigmoid;
    }

    ret.layers[i].num_neurons = layers[i];
    // populate neuron by neuron
    ret.layers[i].neurons =
        (neuron *)calloc(ret.layers[i].num_neurons, sizeof(neuron));
    for (int j = 0; j < layers[i]; j++) {
      neuron n;
      ret.layers[i].neurons[j] = n;
      // populate the weights but not on output layer
      if (i < num_layers - 1) {
        ret.layers[i].neurons[j].num_weights =
            (i < num_layers - 1) ? layers[i + 1] : 0;
        // only calloc if there are weights to be put on heap
        if (ret.layers[i].neurons[j].num_weights) {
          ret.layers[i].neurons[j].weights = (float *)calloc(
              ret.layers[i].neurons[j].num_weights, sizeof(float));
        }
      }
    }
  }
  if (!strcmp(learning_alg, "backprop")) {
    ret.learning_alg = backward_prop;
  }
  return ret;
}

void free_NN(NN *z) {
  for (int i = 0; i < z->num_layers; i++) {
    for (int j = 0; j < z->layers[i].num_neurons - 1; j++) {
      free(z->layers[i].neurons[j].weights);
    }
    free(z->layers[i].neurons);
  }
  free(z->layers);
}

// random right now float (-1,1)
void init_weights(NN *z) {
  // Seed the random number generator
  z->layers[0].neurons[0].weights[0] = 1;
  srand(time(NULL));
  float randFloat;
  for (int i = 0; i < z->num_layers - 1; i++) {
    for (int j = 0; j < z->layers[i].num_neurons; j++) {
      for (int k = 0; k < z->layers[i].neurons[j].num_weights; k++) {
        randFloat = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        z->layers[i].neurons[j].weights[k] = randFloat;
      }
    }
  }
}

// TODO
void train(NN *z, int *inputs, int *expected_outputs, float learning_rate,
           char learning_alg) {
  // assign inputs
  for (int i = 0; i < z->layers[0].num_neurons; i++) {
    z->layers[0].neurons[i].activation = inputs[i];
  }
  // forward prop
  for (int i = 0; i < z->num_layers - 1; i++) {
    forward_prop(&z->layers[i], &z->layers[i + 1]);
  }

  // correct error

  // one iteration done
}
void printLayer(layer *l) {
  for (int i = 0; i < l->num_neurons; i++) {
    printf("NEURON %d \n", i);
    printf("bias: %.3f \n", l->neurons[i].bias);
    printf("activation: %.3f \n", l->neurons[i].activation);
    printf("z: %.3f \n", l->neurons[i].z);
    printf("num_weights: %d \n", l->neurons[i].num_weights);
    printf("weights: [");
    int j;
    for (j = 0; j < l->neurons[i].num_weights - 1; j++) {
      printf("%.3f, ", l->neurons[i].weights[j]);
    }
    if (l->neurons[i].num_weights < 1) {
      printf("]");
    } else {
      printf("%.3f]\n", l->neurons[i].weights[j]);
    }
  }
  printf("\n");
}

void printNN(NN *z) {
  printf("This is my NN \n \n");

  for (int i = 0; i < z->num_layers; i++) {
    printf("THIS IS LAYER %d \n", i);
    printLayer(&z->layers[i]);
    printf("--------------------------- \n");
  }
}

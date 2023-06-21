#include "neural_lib.h"

float sigmoid(float x) { return (1 / (1 + pow(EULER_NUMBER, -x))); }

// optimize we can just use the previous activation
float sigmoid_derivative(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

// lets get branchless
float relu(float x) { return (float)x * (x <= 0); }

float relu_derivative(float x) { return (float)(x > 0); }
// look at all neurons
void forward_prop(layer *feeder, layer *eater) {
  for (int i = 0; i < eater->num_neurons; i++) {
    float sum = 0;
    for (int j = 0; j < feeder->num_neurons; j++) {
      sum += (feeder->neurons[j].weights[i] * feeder->neurons[j].activation);
    }
    eater->neurons[i].net = sum;
    eater->neurons[i].activation = (eater->hidden)
                                       ? relu(sum + eater->neurons[i].bias)
                                       : sigmoid(sum + eater->neurons[i].bias);
  }
}

// TODO
// for sigmoid activation
void backward_prop_output(layer *lay_out, layer *lay_before, float *tv) {

  for (int i = 0; i < lay_out->num_neurons; i++) {
    lay_out->neurons[i].dnet =
        (lay_out->neurons[i].activation - tv[i]) *
        (lay_out->neurons[i].activation * (1 - lay_out->neurons[i].activation));

    for (int j = 0; j < lay_before->num_neurons; j++) {
      lay_before->neurons[j].dweights[i] =
          (lay_out->neurons[i].dnet * lay_before->neurons[j].activation);
      lay_before->neurons[j].dactivation =
          lay_before->neurons[j].weights[j] * lay_out->neurons[i].dnet;
    }
    lay_out->neurons[i].dbias = lay_out->neurons[i].dnet;
  }
}

// TODO
//  relu function
void backward_prop_hidden(layer *lay_out, layer *lay_before) {
  return;

  //   for (int i = 0; i < lay_out->num_neurons; i++) {
  //     lay_out->neurons[i].dnet =
  //         (lay_out->neurons[i].activation - tv[i]) *
  //         (lay_out->neurons[i].activation * (1 -
  //         lay_out->neurons[i].activation));

  //     for (int j = 0; j < lay_before->num_neurons; j++) {
  //       lay_before->neurons[j].dweights[i] =
  //           (lay_out->neurons[i].dnet * lay_before->neurons[j].activation);
  //       lay_before->neurons[j].dactivation =
  //           lay_before->neurons[j].weights[j] * lay_out->neurons[i].dnet;
  //     }
  //     lay_out->neurons[i].dbias = lay_out->neurons[i].dnet;
  //   }
}

// create a delete for my NN
NN Neural_Network(int num_layers, int *layers, char learning_alg) {
  NN ret;
  ret.num_layers = num_layers;
  ret.layers = (layer *)calloc(ret.num_layers, sizeof(layer));
  // populate layer by layer
  for (int i = 0; i < num_layers; i++) {
    layer l;
    ret.layers[i] = l;

    ret.layers[i].num_neurons = layers[i];
    // populate neuron by neuron
    ret.layers[i].neurons =
        (neuron *)calloc(ret.layers[i].num_neurons, sizeof(neuron));
    for (int j = 0; j < layers[i]; j++) {
      neuron n;
      ret.layers[i].neurons[j] = n;
      // populate the weights but not on output layer
      if (i < num_layers - 1) {
        ret.layers[i].hidden = true;
        // TODO initialization of my weights to something other than 0
        // this makes output layer have 0 weight, everything else has next layer
        // amount of neurons as number of weights
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
  ret.learning_alg = learning_alg;
  return ret;
}

void free_NN(NN *net) {
  for (int i = 0; i < net->num_layers; i++) {
    for (int j = 0; j < net->layers[i].num_neurons-1; j++) {
      free(net->layers[i].neurons[j].weights);
    }
    free(net->layers[i].neurons);
  }
  free(net->layers);
}

// random right now float (-1,1)
void init_weights(NN *net) {
  // Seed the random number generator
  srand(time(NULL));
  float randFloat;
  for (int i = 0; i < net->num_layers - 1; i++) {
    for (int j = 0; j < net->layers[i].num_neurons; j++) {
      for (int k = 0; k < net->layers[i].neurons[j].num_weights; k++) {
        randFloat = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        net->layers[i].neurons[j].weights[k] = randFloat;
      }
    }
  }
}

// TODO
void train(NN *net, int *inputs, int *expected_outputs, float learning_rate,
           char learning_alg) {
  // assign inputs
  for (int i = 0; i < net->layers[0].num_neurons; i++) {
    net->layers[0].neurons[i].activation = inputs[i];
  }
  // forward prop
  for (int i = 0; i < net->num_layers - 1; i++) {
    forward_prop(&net->layers[i], &net->layers[i + 1]);
  }

  // correct error

  // one iteration done
}
void printLayer(layer *l) {
  for (int i = 0; i < l->num_neurons; i++) {
    printf("NEURON %d \n", i);
    printf("bias: %.3f \n", l->neurons[i].bias);
    printf("activation: %.3f \n", l->neurons[i].activation);
    printf("net: %.3f \n", l->neurons[i].net);
    printf("num_weights: %d \n", l->neurons[i].num_weights);
  }
  printf("\n");
}

void printNN(NN *net) {
  printf("This is my NN \n \n");

  for (int i = 0; i < net->num_layers; i++) {
    printf("THIS IS LAYER %d \n", i);
    printLayer(&net->layers[i]);
    printf("--------------------------- \n");
  }
}

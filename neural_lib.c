#include "neural_lib.h"

/*
-------------------------------
         MATH FUNCTIONS
-------------------------------
*/

float sigmoid(float x) { return (1 / (1 + pow(EULER_NUMBER, -x))); }
float dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

float relu(float x) { return (float)x * (x > 0); }
float drelu(float x) { return (float)(x > 0); }

float cost(float x, float y) { return 0.5 * powf(y - x, 2); }
float dcost(float x, float y) { return x - y; }

/*
-------------------------------
    INITIALIZATION/CLEANUP
-------------------------------
*/

NN Neural_Network(int num_layers, int *layers) {
  NN ret;
  ret.num_layers = num_layers;
  ret.layers = (layer *)calloc(ret.num_layers, sizeof(layer));
  // populate layer by layer
  for (int i = 0; i < num_layers; i++) {
    layer l;
    ret.layers[i] = l;
    ret.layers[i].output = (i + 1 == num_layers);
    // set hidden layer activation
    if (!ret.layers[i].output) {
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
        // only calloc if there are (d)weights to be put on heap
        if (ret.layers[i].neurons[j].num_weights) {
          ret.layers[i].neurons[j].weights = (float *)calloc(
              ret.layers[i].neurons[j].num_weights, sizeof(float));
          ret.layers[i].neurons[j].dweights = (float *)calloc(
              ret.layers[i].neurons[j].num_weights, sizeof(float));
        }
      }
    }
  }
  return ret;
}

void free_NN(NN *net) {
  for (int i = 0; i < net->num_layers; i++) {
    for (int j = 0; j < net->layers[i].num_neurons; j++) {
      if (i == net->num_layers - 1)
        break;
      free(net->layers[i].neurons[j].weights);
      free(net->layers[i].neurons[j].dweights);
    }
    free(net->layers[i].neurons);
  }
  free(net->layers);
}

// random float (-1,1)
void init_weights(NN *net) {
  // Seed the random number generator
  net->layers[0].neurons[0].weights[0] = 1;
  srand(time(0));
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

/*
-------------------------------
           LOGIC
-------------------------------
*/

//Big help from this article
//https://medium.com/analytics-vidhya/building-neural-network-framework-in-c-using-backpropagation-8ad589a0752d
void backward_prop(NN *n, float *tv) {
  bool output = true;
  for (int l = n->num_layers - 1; l > 0; l--) {
    layer *layer_before = &n->layers[l - 1];
    layer *layer_after = &n->layers[l];
    // do the output layer first
    float true_val;
    for (int i = 0; i < layer_after->num_neurons; i++) {
      // n_after is the output neuron
      neuron *n_after = &layer_after->neurons[i];
      // dz is different for output
      if (output) {
        true_val = tv[i];
        n_after->dz = dcost(n_after->activation, true_val) *
                      layer_after->dactivation(n_after->z);
        output = false;
      } else {
        n_after->dz =
            n_after->dactivation * layer_after->dactivation(n_after->z);
      }
      // update dweights
      for (int j = 0; j < layer_before->num_neurons; j++) {
        neuron *n_before = &layer_before->neurons[j];
        // dcost/dweight(j) = cost_deriv * dactivation *j.z
        n_before->dweights[i] = n_after->dz * n_before->activation;
        // input layers activation is always correct
        if (i > 1) {
          n_before->dactivation = n_after->dz * n_before->weights[i];
        }
      }
      // dcost/dbias(i) = i.dz
      n_after->dbias = n_after->dz;
    }
  }
}

void forward_prop(NN *n) {
  for (int l = 0; l < n->num_layers - 1; l++) {
    layer *layer_before = &n->layers[l];
    layer *layer_after = &n->layers[l + 1];
    for (int i = 0; i < layer_after->num_neurons; i++) {
      float sum = 0;
      for (int j = 0; j < layer_before->num_neurons; j++) {
        sum += (layer_before->neurons[j].weights[i] *
                layer_before->neurons[j].activation);
      }
      layer_after->neurons[i].z = sum;
      layer_after->neurons[i].activation = layer_after->activation(sum);
    }
  }
}

void update_weights(NN *net, float alpha) {
  for (int i = 0; i < net->num_layers - 1; i++) {
    for (int j = 0; j < net->layers[i].num_neurons; j++) {
      for (int k = 0; k < net->layers[i].neurons[j].num_weights; k++) {
        if (i == net->num_layers - 1)
          break;
        net->layers[i].neurons[j].weights[k] -=
            alpha * net->layers[i].neurons[j].dweights[k];
      }
      net->layers[i].neurons[j].bias -= alpha * net->layers[i].neurons[j].dbias;
    }
  }
}

void train_step(NN *net, float *inputs, float *expected_outputs,
                float learning_rate) {
  // assign inputs
  for (int i = 0; i < net->layers[0].num_neurons; i++) {
    net->layers[0].neurons[i].activation = inputs[i];
  }

  forward_prop(net);
  backward_prop(net, expected_outputs);
  update_weights(net, learning_rate);

  // one iteration done
}

/*
-------------------------------
            LOGGING
-------------------------------
*/

void printLayer(layer *l) {
  for (int i = 0; i < l->num_neurons; i++) {
    printf("NEURON %d \n", i);
    printf("bias: %.3f \n", l->neurons[i].bias);
    printf("activation: %.3f \n", l->neurons[i].activation);
    printf("z: %.3f \n", l->neurons[i].z);
    // printf("num_weights: %d \n", l->neurons[i].num_weights);
    printf("weights: [");
    int j;
    for (j = 0; j < l->neurons[i].num_weights - 1; j++) {
      printf("%.3f, ", l->neurons[i].weights[j]);
    }
    if (l->neurons[i].num_weights < 1) {
      printf("] \n");
    } else {
      printf("%.3f]\n", l->neurons[i].weights[j]);
    }
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

void printOut(NN *net) {
  neuron *n;
  for (int i = 0; i < net->layers[net->num_layers - 1].num_neurons; i++) {
    n = &net->layers[net->num_layers - 1].neurons[i];
    printf("%d", n->num_weights);
    printf("Output Neuron %d: %.4f\n", i, n->activation);
  }
}

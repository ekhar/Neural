#include "neural_lib.h"
#include <string.h>

/*
-------------------------------
         MATH FUNCTIONS
-------------------------------
*/

float sigmoid(float x) { return 1 / (1 + exp(-x)); }
float dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

float relu(float x) { return (float)x * (x > 0); }
float drelu(float x) { return (x > 0) * 1.0f; }

float leaky_relu(float x) { return (x > 0) ? x : 0.02 * x; }
float dleaky_relu(float x) { return (x > 0) ? 1 : 0.02; }

float dtanh(float x) { return 1 - pow(tanhf(x), 2); }

float cost(float x, float y) { return pow(x - y, 2); }
float dcost(float x, float y) { return 2 * (x - y); }

/*
-------------------------------
    INITIALIZATION/CLEANUP
-------------------------------
*/

void set_layer_activation(layer *l, char activation_name[]){
  
      l->activation_name = malloc(sizeof(char) * (strlen(activation_name)+1));
      strcpy(l->activation_name, activation_name);
      if (strcmp(activation_name, "leakyRelu") == 0) {
        l->activation = leaky_relu;
        l->dactivation = dleaky_relu;
      }

      else if (strcmp(activation_name, "relu") == 0) {
        l->activation = relu;
        l->dactivation = drelu;
      }

      else if (strcmp(activation_name, "sigmoid") == 0) {
        l->activation = sigmoid;
        l->dactivation = dsigmoid;
      }

      else if (strcmp(activation_name, "tanh") == 0) {
        l->activation = tanhf;
        l->dactivation = dtanh;
      } else {
        fprintf(stderr,
                "activation_name: %s Please specify activation_name as either leakyRelu, "
                "relu, sigmoid, or tanh",
                activation_name);
      }
}
NN Neural_Network(int num_layers, int *layers, char hidden[], char output[]) {
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
      set_layer_activation(&ret.layers[i], hidden);
    }
    // output layer
    else {
      set_layer_activation(&ret.layers[i], output );
      }

    ret.layers[i].num_neurons = layers[i];
    ret.layers[i].neurons =
        (neuron *)calloc(ret.layers[i].num_neurons, sizeof(neuron));

    for (int j = 0; j < layers[i]; j++) {
      neuron n = {0};
      ret.layers[i].neurons[j] = n;
      if (ret.layers[i].activation == sigmoid) {
        ret.layers[i].neurons[j].bias = 0.5;
      }
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
    free(net->layers[i].activation_name);
  }
  free(net->layers);
}

// random float (-1,1)
void init_weights(NN *net) {
  // Seed the random number generator
  net->layers[0].neurons[0].weights[0] = 1;
  srand(time(0));
  float randfloat;
  for (int i = 0; i < net->num_layers - 1; i++) {
    for (int j = 0; j < net->layers[i].num_neurons; j++) {
      for (int k = 0; k < net->layers[i].neurons[j].num_weights; k++) {
        //randfloat = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        //net->layers[i].neurons[j].weights[k] = randfloat;

        net->layers[i].neurons[j].weights[k] =
            sqrt(2.0 / net->layers[i].neurons[j].num_weights) *
            ((float)rand() / (float)RAND_MAX - 0.5); // He initialization
      }
    }
  }
}

/*
-------------------------------
           LOGIC
-------------------------------
*/

// Big help from this article
// https://medium.com/analytics-vidhya/building-neural-network-framework-in-c-using-backpropagation-8ad589a0752d

void backward_prop(NN *net, float *tv) {
  int i, j, k;

  // Output Layer
  for (j = 0; j < net->layers[net->num_layers - 1].num_neurons; j++) {
    layer *before = &net->layers[net->num_layers - 2];
    layer *output = &net->layers[net->num_layers - 1];
    output->neurons[j].dz = dcost(output->neurons[j].activation, tv[j]) *
                            output->dactivation(output->neurons[j].z);

    for (k = 0; k < before->num_neurons; k++) {
      before->neurons[k].dweights[j] =
          (output->neurons[j].dz * before->neurons[k].activation);
      before->neurons[k].dactivation =
          before->neurons[k].weights[j] * output->neurons[j].dz;
    }

    output->neurons[j].dbias = output->neurons[j].dz;
  }

  // Hidden Layers
  for (i = net->num_layers - 2; i > 0; i--) {
    layer *after = &net->layers[i];
    layer *before = &net->layers[i - 1];
    for (j = 0; j < after->num_neurons; j++) {
      after->neurons[j].dz = after->neurons[j].dactivation *
                             after->dactivation(after->neurons[j].z);

      for (k = 0; k < before->num_neurons; k++) {
        before->neurons[k].dweights[j] =
            after->neurons[j].dz * before->neurons[k].activation;

        if (i > 1) {
          before->neurons[k].dactivation =
              before->neurons[k].weights[j] * after->neurons[j].dz;
        }
      }

      after->neurons[j].dbias = after->neurons[j].dz;
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
      layer_after->neurons[i].z = sum + layer_after->neurons[i].bias;
      layer_after->neurons[i].activation =
          layer_after->activation(layer_after->neurons[i].z);
    }
  }
}

void update_weights(NN *net, float alpha) {
  for (int i = 0; i < net->num_layers - 1; i++) {
    for (int j = 0; j < net->layers[i].num_neurons; j++) {
      for (int k = 0; k < net->layers[i].neurons[j].num_weights; k++) {
        if (i != net->num_layers - 1) {
          net->layers[i].neurons[j].weights[k] -=
              alpha * net->layers[i].neurons[j].dweights[k];
        }
      }
      net->layers[i].neurons[j].bias -= alpha * net->layers[i].neurons[j].dbias;
    }
  }
}

void train_step(NN *net, float *inputs, float *expected_outputs,
                float learning_rate) {
  set_inputs(net, inputs);
  forward_prop(net);
  backward_prop(net, expected_outputs);
  update_weights(net, learning_rate);
}

float total_error(NN *net, float *tv) {
  float error = 0;
  for (int i = 0; i < net->layers[net->num_layers - 1].num_neurons; i++) {
    error +=
        cost(net->layers[net->num_layers - 1].neurons[i].activation, tv[i]);
  }

  return error;
}

int max_output(NN* net) {
    layer *l = &net->layers[net->num_layers-1];
    int idx_max = 0;
    float val_max = l->neurons[0].activation;
    for(int i = 1; i < l->num_neurons; i++) {
        if(l->neurons[i].activation > val_max) {
            val_max = l->neurons[i].activation;
            idx_max = i;
        }
    }
    return idx_max;
}

void predict(NN *net, float *inputs) {
  set_inputs(net, inputs);
  forward_prop(net);
}
void set_inputs(NN *net, float *ins) {
  for (int i = 0; i < net->layers[0].num_neurons; i++) {
    net->layers[0].neurons[i].activation = ins[i];
  }
}
/*
-------------------------------
            LOGGING
-------------------------------
*/

void printLayer(layer *l) {
  for (int i = 0; i < l->num_neurons; i++) {
    printf("NEURON %d \n", i);
    printf("bias: %.6f \n", l->neurons[i].bias);
    printf("activation: %.6f \n", l->neurons[i].activation);
    printf("z: %.6f \n", l->neurons[i].z);
    printf("num_weights: %d \n", l->neurons[i].num_weights);
    printf("OUTPUT: %d \n", l->output);
    printf("weights: [");
    int j;
    for (j = 0; j < l->neurons[i].num_weights - 1; j++) {
      printf("%.6f, ", l->neurons[i].weights[j]);
    }
    if (l->neurons[i].num_weights < 1) {
      printf("] \n");
    } else {
      printf("%.6f]\n", l->neurons[i].weights[j]);
    }
  }
  printf("\n");
}

void printdLayer(layer *l) {
  for (int i = 0; i < l->num_neurons; i++) {
    printf("NEURON %d \n", i);
    printf("dbias: %.5f \n", l->neurons[i].dbias);
    printf("dactivation: %.5f \n", l->neurons[i].dactivation);
    printf("dz: %.5f \n", l->neurons[i].dz);
    // printf("num_weights: %d \n", l->neurons[i].num_weights);
    printf("dweights: [");
    int j;
    for (j = 0; j < l->neurons[i].num_weights - 1; j++) {
      printf("%.5f, ", l->neurons[i].dweights[j]);
    }
    if (l->neurons[i].num_weights < 1) {
      printf("] \n");
    } else {
      printf("%.5f]\n", l->neurons[i].dweights[j]);
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

void printdNN(NN *net) {
  printf("This is my NN \n \n");

  for (int i = 0; i < net->num_layers; i++) {
    printf("THIS IS LAYER %d \n", i);
    printdLayer(&net->layers[i]);
    printf("--------------------------- \n");
  }
}

void printOut(NN *net) { printLayer(&net->layers[net->num_layers - 1]); }

/*
-------------------------------
            Saving and loading
-------------------------------
*/
void save_nn(NN *network, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Unable to open file for writing.\n");
        return;
    }

    fwrite(&network->num_layers, sizeof(network->num_layers), 1, file);
    for (int i = 0; i < network->num_layers; i++) {
        int name_len = strlen(network->layers[i].activation_name) + 1; // +1 for null terminator
        fwrite(&name_len, sizeof(name_len), 1, file);
        fwrite(network->layers[i].activation_name, sizeof(char), name_len, file);

        fwrite(&network->layers[i].num_neurons, sizeof(network->layers[i].num_neurons), 1, file);
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            fwrite(&network->layers[i].neurons[j].num_weights, sizeof(network->layers[i].neurons[j].num_weights), 1, file);
            fwrite(network->layers[i].neurons[j].weights, sizeof(float), network->layers[i].neurons[j].num_weights, file);
            fwrite(&network->layers[i].neurons[j].bias, sizeof(network->layers[i].neurons[j].bias), 1, file);
        }
    }
    fclose(file);
}

void read_nn(NN *network, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Unable to open file for reading.\n");
        return;
    }

    fread(&network->num_layers, sizeof(network->num_layers), 1, file);
    network->layers = malloc(sizeof(layer) * network->num_layers);
    for (int i = 0; i < network->num_layers; i++) {
        int name_len;
        fread(&name_len, sizeof(name_len), 1, file);
        network->layers[i].activation_name = malloc(sizeof(char) * name_len);
        fread(network->layers[i].activation_name, sizeof(char), name_len, file);

        set_layer_activation(&network->layers[i], network->layers[i].activation_name);
        fread(&network->layers[i].num_neurons, sizeof(network->layers[i].num_neurons), 1, file);
        network->layers[i].neurons = malloc(sizeof(neuron) * network->layers[i].num_neurons);
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            fread(&network->layers[i].neurons[j].num_weights, sizeof(network->layers[i].neurons[j].num_weights), 1, file);
            network->layers[i].neurons[j].weights = malloc(sizeof(float) * network->layers[i].neurons[j].num_weights);
            network->layers[i].neurons[j].dweights = malloc(sizeof(float) * network->layers[i].neurons[j].num_weights);
            fread(network->layers[i].neurons[j].weights, sizeof(float), network->layers[i].neurons[j].num_weights, file);
            fread(&network->layers[i].neurons[j].bias, sizeof(network->layers[i].neurons[j].bias), 1, file);
        }
    }
    fclose(file);
}

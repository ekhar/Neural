#include "neural_lib.h"

/*
-------------------------------
         MATH FUNCTIONS
-------------------------------
*/

float sigmoid(float x) { return 1 / (1 + exp(-x)); }
float dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

float relu(float x) { return (float)x * (x > 0); }
float drelu(float x) { return (float)(x > 0); }

float cost(float x, float y) { return 0.5 * pow(y - x, 2); }
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
      ret.layers[i].activation = sigmoid;
      ret.layers[i].dactivation = dsigmoid;
    } else {
      ret.layers[i].activation = sigmoid;
      ret.layers[i].dactivation = dsigmoid;
    }

    ret.layers[i].num_neurons = layers[i];
    // populate neuron by neuron
    ret.layers[i].neurons =
        (neuron *)calloc(ret.layers[i].num_neurons, sizeof(neuron));
    for (int j = 0; j < layers[i]; j++) {
      neuron n = {0};
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

// Big help from this article
// https://medium.com/analytics-vidhya/building-neural-network-framework-in-c-using-backpropagation-8ad589a0752d
void backward_prop(NN *n, float *tv) {
  for (int l = n->num_layers - 1; l > 0; l--) {
    layer *layer_before = &n->layers[l - 1];
    layer *layer_after = &n->layers[l];
    // do the output layer first
    float true_val;
    for (int i = 0; i < layer_after->num_neurons; i++) {
      // n_after is the output neuron
      neuron *n_after = &layer_after->neurons[i];
      // dz is different for output
      if (l == n->num_layers - 1) {
        true_val = tv[i];
        n_after->dz = dcost(n_after->activation, true_val) *
                      layer_after->dactivation(n_after->z);
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
        if (l > 1) {
          // come back += was missing in the tutorial
          n_before->dactivation += n_after->dz * n_before->weights[i];
        }
      }
      // dcost/dbias(i) = i.dz
      //running sum dbais
      //n_after->dbias = (i==0) ? n_after->dz : (n_after->dz + layer_after->neurons[i-1].dbias);
      n_after->dbias =  n_after->dz;
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
        if (i != net->num_layers - 1){
          net->layers[i].neurons[j].weights[k] -=
              alpha * net->layers[i].neurons[j].dweights[k];
          }
      }
      //net->layers[i].neurons[j].bias -= alpha*net->layers[i].neurons[net->layers[i].num_neurons-1].dbias;
      //net->layers[i].neurons[j].bias -= alpha*net->layers[i].neurons[j].dbias;
    }
  }
}

void train_step(NN *net, float *inputs, float *expected_outputs,
                float learning_rate) {
  // assign inputs
  set_inputs(net,inputs);

  forward_prop(net);
  backward_prop(net, expected_outputs);
  float err = cost(net->layers[net->num_layers-1].neurons[0].activation, expected_outputs[0]);
  //printf("Error %.5f\n",err);
  update_weights(net, learning_rate);

  // one iteration done
}

float total_error(NN*net, float *tv){
  float error = 0;
  for(int i =0; i<net->layers[net->num_layers-1].num_neurons; i++){
    error += cost(net->layers[net->num_layers-1].neurons[i].activation, tv[i]);
  }

  return error;
}

void predict(NN *net, float *inputs) {
  // assign inputs
  set_inputs(net,inputs);

  forward_prop(net);
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
    // printf("activation: %.6f \n", l->neurons[i].activation);
    // printf("z: %.6f \n", l->neurons[i].z);
    // printf("num_weights: %d \n", l->neurons[i].num_weights);
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

void test_init(NN *net) {

  // int layers [3] = {2,2,2};
  // net = Neural_Network(3,layers);
  float weights[4][2] = {{.15, .2}, {.25, .3}, {.4, .45}, {.5, .55}};

  net->layers[0].neurons[0].weights[0] = weights[0][0];
  net->layers[0].neurons[0].weights[1] = weights[1][0];
  net->layers[0].neurons[1].weights[0] = weights[0][1];
  net->layers[0].neurons[1].weights[1] = weights[1][1];

  net->layers[1].neurons[0].bias = 0.0;
  net->layers[1].neurons[1].bias = 0.0;

  net->layers[1].neurons[0].weights[0] = weights[2][0];
  net->layers[1].neurons[0].weights[1] = weights[3][0];
  net->layers[1].neurons[1].weights[0] = weights[2][1];
  net->layers[1].neurons[1].weights[1] = weights[3][1];
  net->layers[1].neurons[0].bias = 0.35;
  net->layers[1].neurons[1].bias = 0.35;

  net->layers[2].neurons[0].bias = 0.6;
  net->layers[2].neurons[1].bias = 0.6;

  // printNN(net);
  // printf("INITIALIZED\n");
}

void test_forward(NN *net, float *inputs) {
  // printdNN(net);
  set_inputs(net,inputs);
  forward_prop(net);
  // printNN(net);
  // printf("FINISHED FORWARD");
}

void test_back(NN *net, float *tv) {
  backward_prop(net, tv);
  update_weights(net, 0.01);
  // printNN(net);
  // printdNN(net);
  // printf("FINISHED BACK");
  return;
}

void set_inputs(NN *net, float *ins){
    for (int i = 0; i < net->layers[0].num_neurons; i++) {
    net->layers[0].neurons[i].activation = ins[i];
  }
}

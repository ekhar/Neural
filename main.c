#include "neural_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  NN my_net;
  int num_layers = 4;
  int neurons[] = {2, 3, 4, 2};
  float xor_data[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
  float xor_data_expected[4] = {0,1,1,0};

  my_net = Neural_Network(num_layers, neurons);
  init_weights(&my_net);
  printNN(&my_net);
  train_step(&my_net, xor_data[0], xor_data_expected, 0.1);
  printNN(&my_net);
  free_NN(&my_net);
}
#include "neural_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(){
  NN my_net;
  int num_layers = 4;
  int neurons[] = {2,3,4,1};
  const char * learning_alg = "backprop";

  my_net = Neural_Network(num_layers, neurons, learning_alg);  
  init_weights(&my_net);
  printNN(&my_net);
  free_NN(&my_net);
}
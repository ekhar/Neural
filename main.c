#include "neural_lib.h"
#include <stdio.h>
#include <stdlib.h>

int main(){
  NN my_net;
  int num_layers = 4;
  int neurons[] = {2,3,4,1};
  char learning_alg = 'c';

  my_net = Neural_Network(num_layers, neurons, learning_alg);  
  printNN(&my_net);
  free_NN(&my_net);
}
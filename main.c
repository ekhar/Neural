#include "neural_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NUM_LAYERS 4

int main() {

  NN my_net;
  int layers [3] = {2,2,2};
  my_net = Neural_Network(3,layers);

  test_init(&my_net);
  test_forward(&my_net);
  test_back(&my_net);
  return 0;
  //initialization
  // NN my_net;
  int neurons[NUM_LAYERS] = {2, 4, 4, 1};
  float learning_rate = 0.01;
  my_net = Neural_Network(NUM_LAYERS, neurons);
  init_weights(&my_net);
  //printNN(&my_net);

  float xor_data[4][2] = {{0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}};
  float xor_data_expected[4][2] = {{0.0},{1.0},{1.0},{0.0}};

  srand(time(NULL));
  int r = rand() % 4;
  //run net
  for(int i =0; i<50000; i++){
    train_step(&my_net, xor_data[r], xor_data_expected[r], learning_rate);
    // if(i%1000== 0)
    //printOut(&my_net);
    }


  for(int i=0; i<4; i++){
    printf("PREDICTION for %f %f \n", xor_data[i][0],xor_data[i][1]);
    predict(&my_net, xor_data[i]);
    printOut(&my_net);
    printf("TV was %f \n", xor_data_expected[i][0]);
  }

  //Cleanup
  free_NN(&my_net);
}
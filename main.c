#include "neural_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NUM_LAYERS 3

void basic_test(){
  
  NN my_net;
  int layers [3] = {2,2,2};
  my_net = Neural_Network(3,layers);

  double tv[2] = {0.01, 0.99};
  test_init(&my_net);
  double ins[2] = {0.05, 0.1};

  for(int i=0; i<50; i++){
    // printNN(&my_net);
    test_forward(&my_net, ins);
    test_back(&my_net, tv);
    // printf("Error %f", total_error(&my_net, tv));
    // printNN(&my_net);
    test_forward(&my_net, ins);
    printf("TOTAL COST %.5f\n", total_error(&my_net, tv ));
    }
}

int main() {

  // basic_test();
  // return 0;

  //initialization
  NN my_net;
  int neurons[NUM_LAYERS] = {2, 5, 1};
  double learning_rate = 0.01;
  my_net = Neural_Network(NUM_LAYERS, neurons);
  init_weights(&my_net);
  //printNN(&my_net);

  double xor_data[4][2] = {{0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}};
  double xor_data_expected[4][2] = {{0.0},{1.0},{1.0},{0.0}};

  //test_init(&my_net);
  srand(time(NULL));
  int r = rand() % 4;
  //run net
  for(int i =0; i<50000; i++){
    //printf("---------------TRAINING STEP %d ----------------",i);
    train_step(&my_net, xor_data[r], xor_data_expected[r], learning_rate);
    forward_prop(&my_net);
    // test_forward(&my_net, xor_data[i%4]);
    if(i%1000 == 0)
    printf("TOTAL COST %.5f\n", total_error(&my_net, xor_data_expected[r]));
    // if(i%1000=[]= 0)
    // printNN(&my_net);
    // printdNN(&my_net);
    }
  for(int i =0; i<4; i++){
    set_inputs(&my_net, xor_data[i]);
    //forward_prop(&my_net);
    //forward_prop(&my_net);
    printf("THE TRUE VALUE %f \n", xor_data_expected[i][0]);
    printf("TOTAL COST %.5f\n", total_error(&my_net, xor_data_expected[i]));
    printOut(&my_net);
    }
    

  //Cleanup
  free_NN(&my_net);
}


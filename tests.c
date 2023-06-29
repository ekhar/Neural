#include "tests.h"

void test_init(NN *net) {

  int layers[] = {2, 2, 2};
  *net = Neural_Network(3, layers, "leakyRelu", "tanh");

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
}

void test_forward(NN *net, float *inputs) {
  set_inputs(net, inputs);
  forward_prop(net);
}

void test_back(NN *net, float *tv) {
  backward_prop(net, tv);
  update_weights(net, 0.01);
  return;
}

void test_basic() {

  NN my_net;
  int layers[] = {2, 2, 2};
  my_net = Neural_Network(3, layers, "leakyRelu", "tanh");

  float tv[2] = {0.01, 0.99};
  test_init(&my_net);
  float ins[2] = {0.05, 0.1};

  for (int i = 0; i < 50; i++) {
    test_forward(&my_net, ins);
    test_back(&my_net, tv);
    test_forward(&my_net, ins);
    printf("TOTAL COST %.5f\n", total_error(&my_net, tv));
  }
  free_NN(&my_net);
}

void test_readwrite(){
    // Create and initialize the network
    NN network;
    test_init(&network);
    
    // Save the network to a file
    save_nn(&network, "network_test.bin");
    
    // Create a new network and load the saved data
    NN loaded_network;
    read_nn(&loaded_network, "network_test.bin");
    
    // Verify the loaded network
    assert(network.num_layers == loaded_network.num_layers);
    for (int i = 0; i < network.num_layers; i++) {
        assert(network.layers[i].num_neurons == loaded_network.layers[i].num_neurons);
        assert(strcmp(network.layers[i].activation_name, loaded_network.layers[i].activation_name) ==0);
        assert(network.layers[i].activation == loaded_network.layers[i].activation);
        assert(network.layers[i].dactivation == loaded_network.layers[i].dactivation);
        for (int j = 0; j < network.layers[i].num_neurons; j++) {
            assert(network.layers[i].neurons[j].num_weights == loaded_network.layers[i].neurons[j].num_weights);
            for (int k = 0; k < network.layers[i].neurons[j].num_weights; k++) {
                assert(network.layers[i].neurons[j].weights[k] == loaded_network.layers[i].neurons[j].weights[k]);
            }
            assert(network.layers[i].neurons[j].bias == loaded_network.layers[i].neurons[j].bias);
        }
    }
    
    // Clean up
    // Don't forget to write a function to free the memory allocated for your networks
    free_NN(&network);
    free_NN(&loaded_network);
  printf("Readwrite works");
  }

void test_xor() {
  // initialization
  NN my_net;
  char hidden[] = "leakyRelu";
  char output[] = "tanh";
  int layers[] = {2, 4, 4, 1};
  float learning_rate = 0.01;
  my_net = Neural_Network(4, layers, hidden, output);
  init_weights(&my_net);

  float xor_data[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  float xor_data_expected[4][2] = {{0.0}, {1.0}, {1.0}, {0.0}};

  srand(time(NULL));
  int r = 0;
  // run net
  for (int i = 0; i < 50000; i++) {
    int r = rand() % 4;
    train_step(&my_net, xor_data[r], xor_data_expected[r], learning_rate);
    forward_prop(&my_net);
    if (i % 1000 == 0) {
      printf("TOTAL COST %.5f\n", total_error(&my_net, xor_data_expected[r]));
    }
  }
  printNN(&my_net);
  save_nn(&my_net, "net_save_xor");
  NN new_net;
  read_nn(&new_net, "net_save_xor");
  //printf("LOADING ----------------------");
  printNN(&my_net);
  for (int i = 0; i < 4; i++) {
    set_inputs(&my_net, xor_data[i]);
    forward_prop(&my_net);
    printf("THE TRUE VALUE %f \n", xor_data_expected[i][0]);
    printOut(&my_net);
  }
  printf("I AM ALL DONE");

  // Cleanup
  free_NN(&my_net);
}


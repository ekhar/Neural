#include "tests.h"

void test_init(NN *net) {

  double weights[4][2] = {{.15, .2}, {.25, .3}, {.4, .45}, {.5, .55}};

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

void test_forward(NN *net, double *inputs) {
  set_inputs(net, inputs);
  forward_prop(net);
}

void test_back(NN *net, double *tv) {
  backward_prop(net, tv);
  update_weights(net, 0.01);
  return;
}

void set_inputs(NN *net, double *ins) {
  for (int i = 0; i < net->layers[0].num_neurons; i++) {
    net->layers[0].neurons[i].activation = ins[i];
  }
}

void basic_test() {

  NN my_net;
  int layers[3] = {2, 2, 2};
  double tv[2] = {0.01, 0.99};
  double ins[2] = {0.05, 0.1};
  my_net = Neural_Network(3, layers);
  test_init(&my_net);

  for (int i = 0; i < 50; i++) {
    test_forward(&my_net, ins);
    test_back(&my_net, tv);
    test_forward(&my_net, ins);
    printf("TOTAL COST %.5f\n", total_error(&my_net, tv));
  }
  free_NN(&my_net);
}

void xor_test() {
  NN my_net;
  int neurons[4] = {2, 4, 4, 1};
  double learning_rate = 0.05;
  my_net = Neural_Network(4, neurons);
  init_weights(&my_net);

  double xor_data[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double xor_data_expected[4][2] = {{0.0}, {1.0}, {1.0}, {0.0}};

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
  for (int i = 0; i < 4; i++) {
    set_inputs(&my_net, xor_data[i]);
    forward_prop(&my_net);
    // forward_prop(&my_net);
    printf("THE TRUE VALUE %f \n", xor_data_expected[i][0]);
    // printf("TOTAL COST %.5f\n", total_error(&my_net, xor_data_expected[i]));
    printOut(&my_net);
  }

  // Cleanup
  free_NN(&my_net);
}

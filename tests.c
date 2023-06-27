#include "tests.h"

int main(){
  test_xor();
}
void test_init(NN *net) {

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
}

void test_xor() {
  // initialization
  NN my_net;
  char hidden[] = "relu";  
  char output[] = "sigmoid";  
  int layers[] = {2, 4, 4, 1};
  float learning_rate = 0.00;
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
  for (int i = 0; i < 4; i++) {
    set_inputs(&my_net, xor_data[i]);
    forward_prop(&my_net);
    printf("THE TRUE VALUE %f \n", xor_data_expected[i][0]);
    printOut(&my_net);
  }

  // Cleanup
  free_NN(&my_net);
}
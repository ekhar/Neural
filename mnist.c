#include "mnist.h"
#include "neural_lib.h"

int read_arrays(FILE **file, float label[], float data[]) {
  if (*file == NULL) {
    printf("Cannot open file\n");
    return -1;
  }

  int total_data;

  // read header information
  fscanf(*file, "%d\n", &total_data);
  // read labels and data
  for (int i = 0; i < total_data; ++i) {
    // read label
    for (int j = 0; j < 10; ++j) {
      fscanf(*file, "%f\n", &label[j]);
    }
    // read data
    for (int j = 0; j < 28 * 28; ++j) {
      fscanf(*file, "%f\n", &data[j]);
    }
  }
  return total_data;
}

void train_mnist(NN *net, float alpha) {
  float label[10];
  float data[28 * 28];

  FILE *file = fopen("train_data.txt", "r");
  if (file == NULL) {
    printf("Failed to open the file.\n");
    return;
  }

  int total_data;

  for (int i = 0; i < 60000; ++i) {
  total_data = read_arrays(&file, label, data);
    train_step(net, data, label, alpha);
    // save net on every 1000
    if (i % 1000 == 0) {
      printf("Trained %d\n", i);
      save_nn(net, "net_mnist.net");
      printf("Error %f\n", total_error(net, label));
    }
  }
  fclose(file);
}

void test_mnist(NN *net) {
  float label[10];
  float data[28 * 28];
  
  FILE *file = fopen("test_data.txt", "r");
  if (file == NULL) {
    printf("Failed to open the file.\n");
    return;
  }

  int correct = 0;
  int total_data;

  for (int i = 0; i < 10000; ++i) {
    total_data = read_arrays(&file, label, data);
    predict(net, data);
    float curr_max = -9999;
    float temp;
    int max_index = 0;
    for (int i = 0; i < 10; ++i) {
      temp = net->layers[net->num_layers - 1].neurons[i].activation;
      if (temp > curr_max) {
        max_index = i;
        curr_max = temp;
      }
    }

    correct += label[max_index];
  }
  fclose(file);

  printf("THIS IS THE PERCENT CORRECT: %.5f\n", (float)correct / total_data);
}

int main() {
  int layers[] = {28*28,128,10};
  NN mnist_net = Neural_Network(3,layers ,"leakyRelu" ,"tanh");
  init_weights(&mnist_net);
  //NN mnist_net;
  //read_nn(&mnist_net, "net_mnist.net");
   for (int epoch = 1; epoch < 50; ++epoch) {
     train_mnist(&mnist_net, 0.03);
     save_nn(&mnist_net, "net_mnist.net");
     printf("epoch %d\n", epoch);

   }
  save_nn(&mnist_net, "net_mnist.net");
  test_mnist(&mnist_net);
  return 0;
}

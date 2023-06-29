#include "mnist.h"
#include "neural_lib.h"

void one_hot_encode(int input, float output[10]) {
  // Ensure the array is cleared (all zeros)
  memset(output, 0, 10 * sizeof(float));

  // Set the one hot index to 1
  if (input >= 0 && input < 10) {
    output[input] = 1.0f;
  } else {
    printf("Input out of range.\n");
  }
}
// Function to read MNIST data

void read_mnist(char *images_file_path, char *labels_file_path,
                float images[][784], int labels[]){
    FILE *images_file = fopen(images_file_path, "rb");
    FILE *labels_file = fopen(labels_file_path, "rb");

    if (images_file == NULL || labels_file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    // Read the headers
    unsigned char buffer[4];
    fseek(images_file, 4, SEEK_SET);
    fread(buffer, sizeof(unsigned char), 4, images_file);
    int num_items = (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];

    fseek(images_file, 16, SEEK_SET);
    fseek(labels_file, 8, SEEK_SET);

    // Loop over the images and labels
    for (int i = 0; i < num_items; i++) {
        for (int j = 0; j < 784; j++) {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, images_file);
            (*images)[i * 784 + j] = pixel / 255.0f;
        }

        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, labels_file);
        labels[i] = label;
    }

    fclose(images_file);
    fclose(labels_file);
}

void train_mnist(char *images_file_path, char *labels_file_path, NN *net) {
  // Create buffers for the images and labels
  float (*images)[784] = malloc(60000 * sizeof(*images));
  int *labels = malloc(60000 * sizeof(int));
  float output[10];

  // Read the training data
  read_mnist(images_file_path, labels_file_path, images, labels);

  // Loop over the data and train the network
  for (int i = 0; i < 60000; i++) {
    one_hot_encode(labels[i], output);
    train_step(net, images[i], output, 0.03);
      if(i%1000 ==0){
        printf("%d trained\n", i);
      }
  }
  free(images);
  free(labels);
}

void test_mnist(char *images_file_path, char *labels_file_path, NN *net) {
  // Create buffers for the images and labels
  float (*images)[784] = malloc(10000 * sizeof(*images));
  int *labels = malloc(10000 * sizeof(int));

  // Read the testing data
  read_mnist(images_file_path, labels_file_path, images, labels);

  // Loop over the data and test the network
  int correct_predictions = 0;
  for (int i = 0; i < 10000; i++) {
    // TODO: Define the predict function to predict the output of your network
    // for a single example
    predict(net, images[i]);
    int prediction = max_output(net);
    if (prediction == labels[i]) {
      correct_predictions++;
    }
  }

  // Print the accuracy
  float accuracy = (float)correct_predictions / 10000.0;
  printf("Test accuracy: %.2f%%\n", accuracy * 100);

  free(images);
  free(labels);
}

int main() {
  int layers[] = {28 * 28, 128, 10};
  // NN mnist_net = Neural_Network(3, layers, "leakyRelu", "tanh");
  // init_weights(&mnist_net);
  NN mnist_net;
  read_nn(&mnist_net, "net_mnist.net");
  int cap = 0;
  for (int epoch = 0; epoch < cap; ++epoch) {
    train_mnist("mnist/train-images-idx3-ubyte",
               "mnist/train-labels-idx1-ubyte", &mnist_net);
    save_nn(&mnist_net, "net_mnist.net");
    printf("epoch %d\n", epoch);
  test_mnist("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte",
             &mnist_net);
  }
  test_mnist("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte",
             &mnist_net);
  return 0;
}

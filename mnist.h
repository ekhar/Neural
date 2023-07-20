#ifndef TESTS_H
#define TESTS_H

#include "neural_lib.h"
#include <stdint.h> // for uint8_t
#include <stdio.h>

#define IMAGE_SIZE 28

typedef struct MNISTData {
  uint8_t label;
  float float_label[10];
  uint8_t image[IMAGE_SIZE][IMAGE_SIZE];
  float float_image[IMAGE_SIZE][IMAGE_SIZE];
} MNISTData;

void read_images(char *images_file_path, char *labels_file_path,  
                 float images[][784], int labels[]);

void train_mnist(char *images_file_path, char *labels_file_path, NN *net);

void test_mnist(char *images_file_path, char *labels_file_path, NN *net);

void one_hot_encode(int input, float output[10]);

void read_mnist(char *images_file_path, char *labels_file_path,
                float images[][784], int labels[]);

void predict(NN *net, float *input);

int max_output(NN *net);

void train_NN(NN *net);

#endif

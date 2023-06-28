#ifndef TESTS_H
#define TESTS_H

#include <stdio.h>
#include "neural_lib.h"
#include <stdint.h>  // for uint8_t 

#define IMAGE_SIZE 28    

typedef struct MNISTData {
    uint8_t label;
    float float_label[10];
    uint8_t image[IMAGE_SIZE][IMAGE_SIZE];
    float float_image[IMAGE_SIZE][IMAGE_SIZE];
} MNISTData;

void load_mnist();
void train_mnist(NN *net, float alpha);
void test_mnist(NN *net);
int read_arrays(FILE ** file, float label[], float data[]);

#endif

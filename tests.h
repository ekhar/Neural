#include "neural_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NUM_LAYERS 4

void test_init(NN *net);
void test_forward(NN *net, double *inputs);
void test_back(NN *net, double *tv);
void basic_test();
void xor_test();


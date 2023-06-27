#ifndef TESTS_H
#define TESTS_H

#include "neural_lib.h"

void test_init(NN *net);
void test_forward(NN *net, float *inputs);
void test_back(NN *net, float *tv);
void test_basic();
void test_xor();

#endif

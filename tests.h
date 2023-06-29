#ifndef TESTS_H
#define TESTS_H
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "neural_lib.h"

void test_init(NN *net);
void test_forward(NN *net, float *inputs);
void test_back(NN *net, float *tv);
void test_basic();
void test_xor();
void test_readwrite();
//mnist
void load_mnist();
void train_mnist();
void test_mnist(NN *net);

#endif

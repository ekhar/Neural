#ifndef DRAW_H
#define DRAW_H
#define SCALE_FACTOR 20
#include <raylib.h>
#include "neural_lib.h"
#include "mnist.h"

void vizualize_picture(MNISTData *data, char *caption);
void vizualize_net(NN *net);
void predict_picture(NN *net, MNISTData *data);
MNISTData user_draw();
void test_user(NN *net, MNISTData *m);
#endif

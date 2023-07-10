#ifndef DRAW_H
#define DRAW_H
#define SCALE_FACTOR 20
#include <raylib.h>
#include "neural_lib.h"
#include "mnist.h"

void vizualize_picture(MNISTData *data);
void vizualize_net(NN *net);
void predict_picture(NN *net, MNISTData *data);
void user_draw();
#endif

#ifndef DRAW_H
#define DRAW_H
#define SCALE_FACTOR 10
#include <raylib.h>
#include "neural_lib.h"
#include "mnist.h"

void vizualize_picture(MNISTData *data);
void vizualize_net(NN *net);

#endif

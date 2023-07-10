#include "draw.h"
#include "mnist.h"
#include "neural_lib.h"
#include "tests.h"
#include <raylib.h>
int main(){
  //test_readwrite();
  // test_xor();
  //test_basic();
  NN net;
  MNISTData m;
  // read_nn(&net,"saved_nets/xor.net");
  read_nn(&net,"saved_nets/91-86_mnist.net");
  //vizualize_net(&net);
  m = user_draw();
  test_user(&net,&m); 
  return 0;

}

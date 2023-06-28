#include "mnist.h"

int read_arrays(FILE ** file, float label[], float data[]) {
    if (*file == NULL) {
        printf("Cannot open file\n");
        return -1;
    }

    int total_data;

    // read header information

    // read labels and data
    for (int i = 0; i < total_data; ++i) {
        // read label
        for (int j = 0; j < 10; ++j) {
            fscanf(*file, "%f\n", &label[j]);
        }
        // read data
        for (int j = 0; j < 28*28; ++j) {
            fscanf(*file, "%f\n", &data[j]);
        }
    }
  return total_data;
}



void train_mnist(NN *net, float alpha){
  float label[10];
  float data[28*28];
  FILE *file = fopen("mnist/train_data.txt", "r");
  int total_data;
  int label_size;
  int data_size;
  fscanf(file, "%d\n%d\n%d\n", &total_data, &label_size, &data_size);

  for(int i=0; i<total_data;++i){
    total_data = read_arrays(&file,label, data);
    train_step(net, data, label, alpha);
    printf("%d done \n", i);
    }
  fclose(file);
    
}

void test_mnist(NN *net){
  float label[10];
  float data[28*28];
  FILE *file = fopen("mnist/test_data.txt", "r");
  int total_data;
  int label_size;
  int data_size;
  int correct = 0;
  fscanf(file, "%d\n%d\n%d\n", &total_data, &label_size, &data_size);

  for(int i=0; i<total_data;++i){
    total_data = read_arrays(&file,label, data);
    predict(net, data);
    float curr_max = -9999;
    float temp;
    int max_index = 0;
    for(int i=0; i<10; ++i){
      temp = net->layers[net->num_layers-1].neurons[i].activation;
      if(temp>curr_max){
        max_index = i;
        curr_max = temp;
      }  
    }

    correct += label[max_index];
  }
  fclose(file);

  printf("THIS IS THE PERCENT CORRECT: %.5f", (float) correct/total_data);

  
    
}

int main(){
  int layers[] = {28*28,128,10};
  NN mnist_net = Neural_Network(3,layers ,"leakyRelu" ,"tanh");
  train_mnist(&mnist_net, 0.01 );
  return 1;
  }

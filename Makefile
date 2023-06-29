# -*- MakeFile -*-
CFLAGS := -Wall -Wextra -I$(RAYLIB_H)
LDFLAGS := -L$(RAYLIB_LIBS) -lraylib

#Exec
Neural.out: main.o neural_lib.o tests.o
	gcc -O2 main.o neural_lib.o tests.o -o Neural.out -l m

MNIST: mnist.o neural_lib.o 
	gcc -O2 mnist.o neural_lib.o -o MNIST.out -l m

Tests.out: tests.o neural_lib.o 
	gcc -O2 tests.o neural_lib.o -o Tests.out -l m

LATER:
	gcc -O2 main.o mnist.o neural_lib.o -o mnist.out -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

DEBUG:
	gcc -O2 mnist.c neural_lib.c -o debug.out -l m
	
#Dependecies
main.o: main.c neural_lib.h
	gcc -O2 -c main.c

mnist.o: mnist.c mnist.h neural_lib.h
	gcc -O2 -c mnist.c 

neural_lib.o: neural_lib.c neural_lib.h
	gcc -O2 -c neural_lib.c

tests.o: tests.c tests.h
	gcc -O2 -c tests.c

clean:
	rm -f *.o Neural.debug *.out *.net

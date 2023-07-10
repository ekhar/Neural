# -*- MakeFile -*-
CFLAGS := -Wall -Wextra -I$(RAYLIB_H)
LDFLAGS := -L$(RAYLIB_LIBS) -lraylib

#Exec
Neural.out: main.o neural_lib.o tests.o draw.o mnist.o
	clang -g main.o mnist.o neural_lib.o draw.o -o Neural.out -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

MNIST: mnist.o neural_lib.o 
	clang -g mnist.o neural_lib.o -o MNIST.out -l m

Tests.out: tests.o neural_lib.o 
	clang -g tests.o neural_lib.o -o Tests.out -l m

LATER:
	clang -g main.o mnist.o neural_lib.o -o mnist.out -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

DEBUG:
	clang -g mnist.c neural_lib.c -o debug.out -l m
	
#Dependecies
main.o: main.c neural_lib.h
	clang -g -c main.c

mnist.o: mnist.c mnist.h neural_lib.h
	clang -g -c mnist.c 

neural_lib.o: neural_lib.c neural_lib.h
	clang -g -c neural_lib.c

tests.o: tests.c tests.h
	clang -g -c tests.c

draw.o: draw.c draw.h
	clang -g -c draw.c
clean:
	rm -f *.o Neural.debug *.out *.net

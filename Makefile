# -*- MakeFile -*-

#Exec
Neural.out: main.o neural_lib.o
	clang main.o neural_lib.o -o Neural.out -l m

#Dependecies
main.o: main.c
	clang -c main.c

neural_lib.o: neural_lib.c
	clang -c neural_lib.c

Tests.out: tests.o neural_lib.o
	clang tests.o neural_lib.o -o Tests.out -l m

tests.o: tests.c
	clang -c tests.c

# -*- MakeFile -*-

#Exec
Neural.out: main.o neural_lib.o
	clang main.o neural_lib.o -o Neural.out -l m

#Exec
Neural.debug: main.o neural_lib.o
	clang -g main.o neural_lib.o -o Neural.out -l m

#Dependecies
main.o: main.c neural_lib.h
	clang -c main.c

neural_lib.o: neural_lib.c neural_lib.h
	clang -c neural_lib.c

Tests.out: tests.o neural_lib.o 
	clang tests.o neural_lib.o -o Tests.out -l m

tests.o: tests.c tests.h
	clang -c tests.c

clean:
	rm -f *.o Neural.out Neural.debug Tests.out

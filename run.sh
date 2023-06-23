[ -e core*] && rm core*
[ -e Neural.out] && rm Neural.out
gcc -o Neural.out *.c -lm -g
./Neural.out

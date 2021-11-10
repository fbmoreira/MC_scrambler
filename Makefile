EXEC=congen2
CC=gcc
CFLAGS=-O3 -mavx512f -march=native -fopenmp
CDEBUGFLAGS= -ggdb -mavx512f -march=native -Wall -fsanitize=address -fopenmp

all: $(EXEC)

$(EXEC): 
	$(CC) ccongen2.c -o congen2 $(CFLAGS) 

make debug: congen2debug
	$(CC) ccongen2.c -o congen2debug $(CDEBUGFLAGS)


make clean:
	rm ccongen2

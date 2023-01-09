build = $(wildcard build/*.o)

.PHONY: main

main: main.o shuff.o
	gcc $(build) -o bin/main -lm

main.o:
	gcc -c src/main.c -o build/main.o

shuff.o:
	gcc -c src/shuff.c -o build/shuff.o
bin/out: nomain.c 
	gcc $^ NN.c -I. -lm -o $@ -g

clean:
	rm -rf bin


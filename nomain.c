#define TARGET 2

#include <NN.h>
#include <stdio.h>


int main(void) {
	NN nn;

	const float dataset[][3] = {  /* xor */
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 0}
	};

	printf("dataset test: %f\n", dataset[1][0]);

	unsigned int initset[] = {2, 1};
	nn_init(&nn, 2, initset, 2, SIGMOID);

	printmodel(stdout, &nn);
	printf("mse is: %f\n", MSE(&nn, dataset, 4));

	train(dataset, 4, &nn, 0.01, 0.01, 100*10);   

	printf("mse is: %f\n", MSE(&nn, dataset, 4));

	return 0;

}

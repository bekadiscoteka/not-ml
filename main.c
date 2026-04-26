#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DATASIZE 3 /* size of dataset */
#define RAND ( rand() / RAND_MAX )

typedef struct {
	float w, b;
} Neuron;

float MSE(Neuron *); 
void feed(Neuron *, float, float, unsigned);
float y_calc(Neuron *n, float x) {

	return (n->w * x) + n->b;
}

float dataset[][2] = {
	{2, 4},
	{4, 8},
	{8, 16}
};

void feed(Neuron *n, float eps, float lr, unsigned epoch) {

	float diff;
	Neuron n_new = *n;

	for (unsigned i=0; i < epoch; i++) {
		n_new.w += eps;
		diff = MSE(&n_new) - MSE(n);
		n->w -= lr * diff;  	
		printf("mse: %f\n", MSE(n));
	}

}

float MSE (Neuron *n) {
	float mse = 0;
	for (int i=0; i < DATASIZE; i++) 
		mse += dataset[i][1] - y_calc(n, dataset[i][1]);	
	return mse;
}

int main(void) {
	Neuron n1 = { RAND, RAND };
	float eps, lr;
   	eps = lr = 0.01;	
	feed(&n1, eps, lr, 100);
	printf("16: %f", y_calc(&n1, 16));
	return 0;
}



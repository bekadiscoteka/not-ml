#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RAND ( rand() / RAND_MAX )
#define sigmoid(x) ( 1.0F / ( 1 + exp(-x) ) )

#define DATASIZE 4      /* size of dataset */
#define TARGET 2 	/* index of target */
#define INPUTSIZE 2

typedef struct {
	float w, b;
} Neuron;

float MSE(Neuron *); 
void feed(Neuron *, float, float, unsigned);

float dataset[][3] = { /* and oper */
	{0, 0, 0},
	{1, 0, 0},
	{0, 1, 0},
	{1, 1, 1}
};

float y_calc(Neuron *n, float x[]) {
	float x_sum, y;
	x_sum = 0;	

	for (int j=0; j < INPUTSIZE; j++) 
		x_sum += x[j];					
	y = (n->w * x_sum) + n->b;
	return sigmoid(y);
}

void feed(Neuron *n, float eps, float lr, unsigned epoch) {

	float gradw, gradb;
	Neuron n_neww, n_newb;

	n_neww = n_newb = *n;
	for (unsigned i=0; i < epoch; i++) {
		n_neww = n_newb = *n;
		n_neww.w += eps;
		n_newb.b += eps;

		gradw = (MSE(&n_neww) - MSE(n)) / eps;
		gradb = (MSE(&n_newb) - MSE(n)) / eps;

		n->w -= lr * gradw;  	
		n->b -= lr * gradb;
		// printf("mse: %f\n", MSE(n));
	}

}

float MSE (Neuron *n) {
	float mse = 0;
	for (int i=0; i < DATASIZE; i++) 
		mse += pow(dataset[i][TARGET] - y_calc(n, dataset[i]), 2);	
	return mse;
}

int main(void) {
	Neuron n1 = { RAND, RAND };
	float eps, lr;
   	eps = lr = 0.01;	
	feed(&n1, eps, lr, 100*1000);

	printf("mse: %f\n", MSE(&n1));
	float zero[] = {1, 0};
	float one[] = {1, 1};
	printf("one: %f", y_calc(&n1, one));
	return 0;
}



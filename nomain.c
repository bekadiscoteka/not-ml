#define MAT_IMPLEMENTATION
#define NN_IMPLEMENTATION
#include <NN.h>
#include <time.h>

#include <stdio.h>

float dataset[] = {
	 0, 0, 0 ,
	 0, 1, 1 ,
	 1, 0, 1 ,
	 1, 1, 0 
};



int main(void) {

	srand(time(0));

	Mat m_dataset = {
		.rows = 4,
		.cols = 3,
		.stride = 3,
		.p = dataset
	};

	Mat ti = mat_sharsub(m_dataset, 0, 4, 0, 2);
	Mat to = mat_sharcol(m_dataset, 2);

	NN nn = nn_alloc( (size_t[2]) { 2, 1 }, 2, 2 );
	for (size_t i=0; i<nn.size; i++) {
		mat_rand(nn.w[i], 0, 1);
		mat_rand(nn.b[i], 0, 1);
	}

	printf("cost: %f\n", nn_cost(&nn, ti, to)); 

	NN grad = nn_alloc( (size_t[2]) { 2, 1 }, 2, 2 );

	for (int i=0; i<100; i++) {
		nn_fdiff(&grad, &nn, 0.1f, ti, to);
		nn_train(&nn, &grad, 0.1f);
		
		printf("cost: %f\n", nn_cost(&nn, ti, to)); 
	}

	printf("final cost: %f\n", nn_cost(&nn, ti, to)); 

	return 0;

}

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
	nn_rand(&nn);

	NN_PRINT(&nn);

	printf("cost: %f\n", nn_cost(&nn, ti, to)); 

	NN grad = nn_alloc( (size_t[2]) { 2, 1 }, 2, 2 );

	for (int i=0; i<100*100; i++) {
		nn_fdiff(&grad, &nn, 0.01f, ti, to);
		nn_train(&nn, &grad, 0.1f);
		
		//printf("cost: %f\n", nn_cost(&nn, ti, to)); 
	}

#if 1

	for (size_t i=0; i < m_dataset.rows; i++) {
		MAT_ON_STACK(input, 1, 2);
		MAT_ON_STACK(output, 1, 1);
		input.p[0] = MAT_AT(m_dataset, i, 0);
		input.p[1] = MAT_AT(m_dataset, i, 1);
		NN_SETINPUT(&nn, input); 
		NN_SETOUTPUT(&nn, output);
		nn_forward(&nn);
		printf("%d ^ %d: %f\n", (int) MAT_AT(m_dataset, i, 0), (int) MAT_AT(m_dataset, i, 1), MAT_AT(nn.om, 0, 0));
	}

#endif

	printf("final cost: %f\n", nn_cost(&nn, ti, to)); 

	return 0;

}

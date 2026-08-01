#define MAT_IMPLEMENTATION
#define NN_IMPLEMENTATION
#include <NN.h>
#include <time.h>

#include <stdio.h>




int main(void) {

	srand(time(0));

	float dataset[] = {
		 0, 0, 0 ,
		 0, 1, 1 ,
		 1, 0, 1 ,
		 1, 1, 0 
	};
	Mat mat_dataset = { 
		.rows = 4, 
		.cols = 3, 
		.stride = 3,
		.p = dataset
	};

	size_t arch[3] = {2, 2, 1};
	NN nn = nn_alloc( arch, sizeof(arch)/sizeof(size_t) );
	nn_rand(&nn);
	NN_PRINT(&nn);

	float tp[2] = {0, 0};
	Mat tm = {.cols = 2, .rows = 1, .stride=2, .p=tp};
	mat_cpy(NN_INPUT(&nn), tm);
	nn_forward(&nn);
	MAT_PRINT(NN_OUTPUT(&nn));

	NN g = nn_alloc(arch, sizeof(arch)/sizeof(size_t));
	for (size_t epoch=0; epoch<10; epoch++) {
		for (size_t n=0; n<mat_dataset.rows; n++) {
			Mat x0 = mat_sharsub(mat_dataset,  n, 1, 0, 2);
			Mat y0 = mat_sharsub(mat_dataset, n, 1, 2, 1); 	

			mat_cpy(NN_INPUT(&nn), x0);
			nn_forward(&nn);
			nn_backward(&nn, &g, y0);
			
			nn_train(&nn, &g, 0.01);
		}
	}

	NN_PRINT(&nn);

	printf("TEST CASE");
	mat_cpy(NN_INPUT(&nn), tm);
	nn_forward(&nn);
	MAT_PRINT(NN_OUTPUT(&nn));

	return 0;
}


#define MAT_IMPLEMENTATION
#define NN_IMPLEMENTATION
#include <NN.h>
#include <time.h>

#include <stdio.h>




int main(void) {

	srand(time(0));

	float dataset[] = {

		 0, 0, 0,	0, 0,
		 0, 0, 1,	0, 1,
		 0, 1, 0,	0, 1,
		 0, 1, 1,	1, 0,
		 1, 0, 0,	0, 1,
		 1, 0, 1,	1, 0,
		 1, 1, 0,	1, 0,
		 1, 1, 1,	1, 1

	};

	Mat mat_dataset = { 
		.rows = 8, 
		.cols = 5, 
		.stride = 5,
		.p = dataset
	};

	size_t arch[] = {3, 5, 2};
	NN nn = nn_alloc( arch, sizeof(arch)/sizeof(size_t), mat_dataset.rows );
	nn_rand(&nn);
	NN_PRINT(&nn);

	Mat x = mat_sharsub(mat_dataset, 0, mat_dataset.rows, 0, 3);
	Mat y = mat_sharsub(mat_dataset, 0, mat_dataset.rows, 3, 2);
	printf("-----DATASET-----\n");
	MAT_PRINT(x);
	MAT_PRINT(y);

	mat_cpy(NN_INPUT(&nn), x);
	nn_forward(&nn);
	MAT_PRINT(NN_OUTPUT(&nn));
	NN g = nn_alloc(arch, sizeof(arch)/sizeof(size_t), mat_dataset.rows );

	for (size_t epoch=0; epoch<1000*100; epoch++) {
		mat_cpy(NN_INPUT(&nn), x);
		nn_forward(&nn);
		nn_backward(&nn, &g, y);
		
		nn_train(&nn, &g, 0.01);
	}
	NN_PRINT(&nn);

	printf("TEST CASE");
	MAT_PRINT(NN_OUTPUT(&nn));

	return 0;

	
}


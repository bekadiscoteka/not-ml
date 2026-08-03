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
	NN nn = nn_alloc( arch, sizeof(arch)/sizeof(size_t), mat_dataset.rows );
	nn_rand(&nn);
	NN_PRINT(&nn);

	float tp[2] = {1, 1};
	Mat tm = {.cols = 2, .rows = 1, .stride=2, .p=tp};
	mat_cpy(NN_INPUT(&nn), tm);
	nn_forward(&nn);
	MAT_PRINT(NN_OUTPUT(&nn));
	NN g = nn_alloc(arch, sizeof(arch)/sizeof(size_t), mat_dataset.rows );
#if 1
	Mat x = mat_sharsub(mat_dataset, 0, 4, 0, 2);
	Mat y = mat_sharcol(mat_dataset, 2); 
	for (size_t epoch=0; epoch<500; epoch++) {
		mat_cpy(NN_INPUT(&nn), x);
		nn_forward(&nn);
		nn_backward(&nn, &g, y);
		
		nn_train(&nn, &g, 0.1);
	}
	NN_PRINT(&g);

	printf("TEST CASE");
	mat_cpy(NN_INPUT(&nn), tm);
	nn_forward(&nn);
	MAT_PRINT(NN_OUTPUT(&nn));

	return 0;

#else 

	Mat x0 = mat_sharsub(mat_dataset,  0, 1, 0, 2);
	Mat y0 = mat_sharsub(mat_dataset, 0, 1, 2, 1); 	

	mat_cpy(NN_INPUT(&nn), x0);
	nn_forward(&nn);
	nn_backward(&nn, &g, y0);

	NN_PRINT(&g);

#endif
	
}


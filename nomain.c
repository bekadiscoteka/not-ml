#define MAT_IMPLEMENTATION
#define NN_IMPLEMENTATION
#include <NN.h>

#include <stdio.h>



int main(void) {
	size_t s[] = { 2, 1};
	NN nn = nn_alloc(s, sizeof(s)/sizeof(size_t), 2);

	for (size_t i=0; i < nn.size; i++) {
		mat_rand(nn.w[i], 1.0, 1.0);
		mat_rand(nn.b[i], 1.0, 1.0);
	}

	nn_print(&nn, "nn");

	Mat input = { 
		.rows = 2, 
		.cols = 2,
		.stride = 2,
		.p = (float[]) {1, 1, 2, 2}
	};

	nn.im = mat_share(input);
	nn.om = mat_alloc(2, 1);
	MAT_PRINT(input);
	nn_forward(&nn);

	MAT_PRINT(nn.om);
	
	return 0;
}

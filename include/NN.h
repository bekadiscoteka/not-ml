#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <matrix.h>
#include <math.h>

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif

#ifndef NN_CALLOC 
#define NN_CALLOC calloc
#endif

#ifndef NN_ASSERT
#define NN_ASSERT assert
#endif


typedef struct {
	Mat im;
	Mat *w;
	Mat *b;
	Mat om;
	size_t size;
} NN;

enum nn_alloc {RAND, ZERO};

NN nn_alloc(size_t *arch, size_t size, size_t input_size);
NN *nn_forward(NN *nn);
float nn_cost(const NN *nn, const Mat ti, const Mat to);
NN *nn_fdiff(NN* grad, NN *nn, float eps, const Mat ti, const Mat to);
NN *nn_train(NN *nn, NN* grad, float lr);
void nn_print(NN *nn, const char *name);
float sigmoidf(float);



#endif

#ifdef NN_IMPLEMENTATION


NN nn_alloc(size_t *arch, size_t arch_size, size_t input_size) {
	NN_ASSERT( arch_size > 0 );
	NN nn;
	nn.size = arch_size;

	NN_ASSERT( (nn.w = NN_CALLOC(arch_size, sizeof(Mat))) != NULL );
	NN_ASSERT( (nn.b = NN_CALLOC(arch_size, sizeof(Mat))) != NULL );
	

	for (size_t i = 0; i < arch_size; i++) {
		nn.w[i] = mat_alloc( input_size, arch[i] );		
		nn.b[i] = mat_alloc( 1, arch[i] );
		input_size = arch[i]; 
	}

	return nn;
	
}

void nn_print(NN *nn, const char *name) {
	printf("%s = {\n", name);	
	char buf[256];
	for (size_t i=0; i < nn->size; i++) {
		snprintf(buf, sizeof(buf), "nw-mx: %3zu", i);
		mat_print(nn->w[i], buf, 4);
		snprintf(buf, sizeof(buf), "bs-mx: %3zu", i);
		mat_print(nn->b[i], buf, 4);
	}
	printf("}\n");
}


NN *nn_forward(NN *nn) {
	Mat bufm = mat_alloc(32, 32);
	bufm = mat_cpy(bufm, nn->im);	
	for (size_t i=0; i < nn->size; i++) {
		float tempf[bufm.rows * ( nn->w[i].cols )];
		
		Mat temp = {
			.rows = bufm.rows,
			.cols = nn->w[i].cols,
			.stride = nn->w[i].cols,
			.p = tempf
		};
		
		temp = mat_dot(temp, bufm, nn->w[i]);

		MAT_PRINT(temp);

		bufm.rows = temp.rows;
		bufm.cols = temp.cols;

		bufm = mat_brcst(bufm, temp, nn->b[i]);
		MAT_PRINT(bufm);
	}

	nn->om = mat_cpy(nn->om, bufm);
	return nn;
}





#endif


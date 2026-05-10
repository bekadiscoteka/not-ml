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

#define NN_SETINPUT(nnp, x) {				\
	NN_ASSERT( (nnp)->im.cols == (x).cols );	\
	(nnp)->im = mat_share((x));				\
}

#define NN_SETOUTPUT(nnp, y) {;				\
	NN_ASSERT( (nnp)->om.cols == (y).cols );	\
	(nnp)->om = mat_share( (y) );				\
}

#define NN_PRINT(nn) nn_print(nn, #nn)

#define SIGMOIDF(x) ( (float) ( exp(x) / (1 + exp(x))) )


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
float nn_cost(NN *nn, const Mat ti, const Mat to);
NN *nn_fdiff(NN* grad, NN *nn, float eps, const Mat ti, const Mat to);
NN *nn_train(NN *nn, NN* grad, float lr);
void nn_print(NN *nn, const char *name);
void mat_sigmoid(Mat m);


#endif

#ifdef NN_IMPLEMENTATION


NN nn_alloc(size_t *arch, size_t arch_size, size_t input_size) {
	NN_ASSERT( arch_size > 0 );
	NN nn;
	nn.size = arch_size;

	NN_ASSERT( (nn.w = NN_CALLOC(arch_size, sizeof(Mat))) != NULL );
	NN_ASSERT( (nn.b = NN_CALLOC(arch_size, sizeof(Mat))) != NULL );
	
	nn.im.cols = input_size;

	for (size_t i = 0; i < arch_size; i++) {
		nn.w[i] = mat_alloc( input_size, arch[i] );		
		nn.b[i] = mat_alloc( 1, arch[i] );
		input_size = arch[i]; 
	}

	nn.om.cols = input_size;

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


void mat_sigmoid(Mat m) {
	for (size_t i=0; i < m.rows; i++) {
		for (size_t j=0; j < m.cols; j++) {
			MAT_AT(m, i, j) = SIGMOIDF( (MAT_AT(m, i, j) ) );
		}
	}
}

NN *nn_forward(NN *nn) {
	
	NN_ASSERT( nn->im.rows == nn->om.rows );

	MAT_ON_STACK(bufm, 32, 32);
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

		bufm.rows = temp.rows;
		bufm.cols = temp.cols;

		bufm = mat_brcst(bufm, temp, nn->b[i]);

		mat_sigmoid(bufm);
	}

	nn->om = mat_cpy(nn->om, bufm);
	return nn;
}


float nn_cost(NN *nn, const Mat ti, const Mat to) {
	NN_ASSERT(ti.rows == to.rows);
	
	MAT_ON_STACK( out, to.rows, to.cols );
	
	NN_SETINPUT(nn, ti);
	NN_SETOUTPUT(nn, out);
	nn_forward(nn);

	float sum_batch = 0;
	for (size_t i=0; i < out.rows; i++) {
		float sum_MSE_neuron = 0;
		for (size_t j=0; j < out.cols; j++) {
			float diff = MAT_AT(to, i, j) - MAT_AT(out, i, j);	
			sum_MSE_neuron += diff * diff;	
		}
		sum_batch += sum_MSE_neuron / out.cols;
	}

	return (sum_batch /= out.rows);
}


NN *nn_fdiff(NN* grad, NN *nn, float eps, const Mat ti, const Mat to) {
	NN_ASSERT( grad->size == nn->size );
	float cost = nn_cost(nn, ti, to); 

	for (size_t i=0; i<nn->size; i++) {
		Mat *layer = nn->w + i;
		Mat *grad_layer = grad->w + i;
		for (size_t j=0; j < layer->rows; j++) {
			for (size_t k=0; k < layer->cols; k++) {
				MAT_AT(nn->w[i], j, k) += eps;	
				float cost_eps = nn_cost(nn, ti, to);
				MAT_AT(nn->w[i], j, k) -= eps;	

				MAT_AT(grad->w[i], j, k) = (cost_eps - cost) / eps;	

			}
		}

		for (size_t j=0; j < nn->b[i].cols; j++) {
				MAT_AT(nn->b[i], 0, j) += eps;	
				float cost_eps = nn_cost(nn, ti, to);
				MAT_AT(nn->b[i], 0, j) -= eps;	

				MAT_AT(grad->b[i], 0, j) = (cost_eps - cost) / eps;	
		}
	}

	return grad;
}


NN *nn_train(NN *nn, NN* grad, float lr) {
	for (size_t i=0; i<nn->size; i++) {
		Mat *layer = nn->w + i;
		Mat *grad_layer = grad->w + i;
		for (size_t j=0; j < grad->w[i].rows; j++) {
			for (size_t k=0; k < grad->w[i].cols; k++) {
				MAT_AT(nn->w[i], j, k) -= lr * MAT_AT(grad->w[i], j, k);
			}
		}

		for (size_t j=0; j < grad->b[i].cols; j++) {
			MAT_AT(nn->b[i], 0, j) -= lr * MAT_AT(grad->b[i], 0, j);
		}
	}
	return nn;
}

#endif


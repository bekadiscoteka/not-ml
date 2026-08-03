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

#define NN_INPUT(nnp) ( (nnp)->a[0] )
#define NN_OUTPUT(nnp) ( (nnp)->a[((nnp)->size)-1] )

#define NN_PRINT(nn) nn_print(nn, #nn)

#define SIGMOIDF(x) ( 1.0f / (1.0f + expf(-(x))) )

typedef struct {
	Mat *w;
	Mat *b;
	Mat *z;
	Mat *a;
	size_t size;
} NN;

NN nn_alloc(size_t *arch, size_t size, size_t batch);
NN *nn_forward(NN *nn);
//float nn_cost(NN *nn, const Mat ti, const Mat to);
//NN *nn_fdiff(NN* grad, NN *nn, float eps, const Mat ti, const Mat to);
NN *nn_train(NN *nn, NN* grad, float lr);
void nn_print(NN *nn, const char *name);
void mat_sigmoid(Mat m);
NN *nn_rand(NN* nn);
void nn_backward(NN *nn, NN *g, Mat y);

#endif

#ifdef NN_IMPLEMENTATION


NN nn_alloc(size_t *arch, size_t size, size_t batch) {
	NN_ASSERT( size > 0 );
	NN nn;
	nn.size = size;

	NN_ASSERT( (nn.w = NN_CALLOC(size, sizeof(Mat))) != NULL );
	NN_ASSERT( (nn.b = NN_CALLOC(size, sizeof(Mat))) != NULL );
	NN_ASSERT( (nn.z = NN_CALLOC(size, sizeof(Mat))) != NULL );
	NN_ASSERT( (nn.a = NN_CALLOC(size, sizeof(Mat))) != NULL );

	size_t input_size = 1;
	for (size_t i = 0; i < size; i++) {
		nn.w[i] = mat_alloc( input_size, arch[i] );		
		nn.b[i] = mat_alloc( 1, arch[i] ); 
		nn.z[i] = mat_alloc( batch, arch[i] ); // for only single sample input
		nn.a[i] = mat_alloc( batch, arch[i] ); // for only single sample input
		input_size = arch[i]; 
	}

	return nn;
}

NN *nn_rand(NN *nn) {
	for (size_t i=1; i<nn->size; i++) {
		mat_rand(nn->w[i], 0, 1);
		mat_rand(nn->b[i], 0, 1);
	}
	return nn;
}

void nn_print(NN *nn, const char *name) {
	printf("%s = {\n", name);	
	char buf[256];
	for (size_t i=1; i < nn->size; i++) {
		snprintf(buf, sizeof(buf), "nw-mx: %3zu", i);
		mat_print(nn->w[i], buf, 4);
		snprintf(buf, sizeof(buf), "bs-mx: %3zu", i);
		mat_print(nn->b[i], buf, 4);
		putchar('\n');
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

float sigmoidf(float x) {
	float expx = expf(x);
	return expx / (1.0 + expx);
}

/*
NN *nn_forward_depr(NN *nn) {
	
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
*/

NN *nn_forward(NN *nn) {
	
	for (size_t i=1; i<nn->size; i++) {
		mat_dot(nn->z[i], nn->a[i-1], nn->w[i]);  	
		mat_brcst(nn->z[i], nn->z[i], nn->b[i]);

		for (size_t r=0; r<nn->z[i].rows; r++) 
			for (size_t c=0; c<nn->z[i].cols; c++) 
				MAT_AT( nn->a[i], r, c ) = sigmoidf( MAT_AT(nn->z[i], r, c) );
	}	

	return nn;
}

void nn_backward(NN *nn, NN *g, Mat y) {
	NN_ASSERT( y.cols == NN_OUTPUT(nn).cols );
	NN_ASSERT( y.rows == NN_OUTPUT(nn).rows );

	size_t last = nn->size-1;
	size_t n = y.rows;
	
#define delta ( g->a )

	for (size_t l=nn->size-1; l > 0; l--) {
		// calculate last bias gradient
		MAT_ON_STACK( a_square, nn->a[l].rows, nn->a[l].cols );
		mat_mul( a_square, nn->a[l], nn->a[l] );
		Mat da_dz = mat_subtr( a_square, nn->a[l], a_square );	
		
			if ( l == last ) {
				MAT_ON_STACK( diff, y.rows, y.cols );
				mat_subtr( diff, NN_OUTPUT(nn), y );

				for (size_t _r = 0; _r < diff.rows; _r++) 
					for (size_t _c = 0; _c < diff.cols; _c++) 
						MAT_AT(diff, _r, _c) *= 2;

				Mat dC_da = diff;

				mat_mul( delta[l], dC_da, da_dz );

			} else {
				MAT_ON_STACK( w_T, nn->w[l+1].cols, nn->w[l+1].rows );
				mat_dot( delta[l], delta[l+1], mat_transpose( w_T, nn->w[l+1] ) );
				mat_mul( delta[l], delta[l], da_dz ); 		
			}


		g->b[l] = mat_colmean( g->b[l], delta[l] );

		MAT_ON_STACK( a_T, nn->a[l-1].cols, nn->a[l-1].rows );
		g->w[l] = mat_dot( g->w[l], mat_transpose( a_T, nn->a[l-1] ), delta[l] );

		for (size_t _r=0; _r<g->w[l].rows; _r++) 
			for (size_t _c=0; _c<g->w[l].cols; _c++) 
				MAT_AT(g->w[l], _r, _c) /= n;
		
	}

#undef delta
}


/*

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
*/

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


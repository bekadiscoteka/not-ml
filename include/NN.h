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
#define SIGMOIDF(x) ( (float) ( exp(x) / (1 + exp(x))) )
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


	for (size_t l=nn->size-1; l > 0; l--) {
		// calculate last bias gradient
		MAT_ON_STACK( z_square, nn->z[l].rows, nn->z[l].cols );
		mat_mul( z_square, nn->z[l], nn->z[l] );
		Mat da_dz = mat_subtr( z_square, nn->z[l], z_square );	
		
			if ( l == last ) {
				MAT_ON_STACK( diff, y.rows, y.cols );
				mat_subtr( diff, NN_OUTPUT(nn), y );

				for (size_t _r = 0; _r < diff.rows; _r++) 
					for (size_t _c = 0; _c < diff.cols; _c++) 
						MAT_AT(diff, _r, _c) = MAT_AT(diff, _r, _c) * 2;

				Mat dC_da = diff;

				Mat dC_db = mat_mul( dC_da, dC_da, da_dz );
				MAT_ON_STACK( sum_dC_db, 1, dC_db.cols );

				{
					MAT_ON_STACK( ident_mat, 1, dC_db.rows );
					mat_fill( ident_mat, 1 );
					mat_dot( sum_dC_db, ident_mat, dC_db );
				}

				
				for (size_t _r = 0; _r < sum_dC_db.rows; _r++) 
					for (size_t _c = 0; _c < sum_dC_db.cols; _c++) 
						MAT_AT(sum_dC_db, _r, _c) = MAT_AT(sum_dC_db, _r, _c) / n;

				Mat avg_dC_db = sum_dC_db;

				mat_cpy(g->b[l], avg_dC_db);

			} else {
				mat_fill(g->b[l], 0);
				for (size_t i=0; i < da_dz.rows; i++) {
					Mat z_row = mat_sharrow(da_dz, i);
					MAT_ON_STACK( w_T, nn->w[l+1].cols, nn->w[l+1].rows );
					mat_transpose( w_T, nn->w[l+1] );

					for (size_t r=0; r<w_T.rows; r++) 
						mat_mul(mat_sharrow(w_T, r), z_row, mat_sharrow(w_T, r));
					Mat dz_db = w_T;					

					MAT_ON_STACK(sum_dz_db, 1, dz_db.cols);
					{
						MAT_ON_STACK( ident_mat, 1, dz_db.rows );
						mat_fill( ident_mat, 1 );
						mat_dot( sum_dz_db, ident_mat, dz_db );
					}
					
					for (size_t _r = 0; _r < sum_dz_db.rows; _r++) 
						for (size_t _c = 0; _c < sum_dz_db.cols; _c++) 
							MAT_AT(sum_dz_db, _r, _c) = MAT_AT(sum_dz_db, _r, _c) / dz_db.rows;

					Mat avg_dz_db = sum_dz_db;

					MAT_ON_STACK( dC_db, g->b[l+1].cols, avg_dz_db.cols );
					for (size_t c=0; c<g->b[l+1].cols; c++) {
						MAT_ON_STACK( temp, 1, avg_dz_db.cols );
						mat_cpy(temp, avg_dz_db);

						for (size_t _r = 0; _r < temp.rows; _r++) 
							for (size_t _c = 0; _c < temp.cols; _c++) 
								MAT_AT(temp, _r, _c) = MAT_AT(temp, _r, _c) * MAT_AT(g->b[l+1], 0, c);

						mat_cpy(mat_sharrow(dC_db, c), temp);
					}
					MAT_ON_STACK(sum_dC_db, 1, dC_db.cols);
					{
						MAT_ON_STACK( ident_mat, 1, dC_db.rows );
						mat_fill( ident_mat, 1 );
						mat_dot( sum_dC_db, ident_mat, dC_db );
					}

					for (size_t _r = 0; _r < sum_dC_db.rows; _r++) 
						for (size_t _c = 0; _c < sum_dC_db.cols; _c++) 
							MAT_AT(sum_dC_db, _r, _c) = MAT_AT(sum_dC_db, _r, _c) / g->b[l+1].cols;

					Mat avg_dC_db = sum_dC_db;

					mat_add( g->b[l], g->b[l], avg_dC_db );
				}
				
				for (size_t _r = 0; _r < g->b[l].rows; _r++) 
					for (size_t _c = 0; _c < g->b[l].cols; _c++) 
						MAT_AT(g->b[l], _r, _c) = MAT_AT(g->b[l], _r, _c) / n;

			}

		// calculate l weight gradient
		MAT_ON_STACK( a_prev_clone, nn->a[l-1].rows, nn->a[l-1].cols );
		mat_cpy( a_prev_clone, nn->a[l-1] );
		for (size_t i=0; i<g->b[l].cols; i++) {
				
			for (size_t _r = 0; _r < a_prev_clone.rows; _r++) 
				for (size_t _c = 0; _c < a_prev_clone.cols; _c++) 
					MAT_AT(a_prev_clone, _r, _c) = MAT_AT(a_prev_clone, _r, _c) * MAT_AT(g->b[l], 0, i);

			// l means "for local neuron"
			Mat dw_l = a_prev_clone;
			MAT_ON_STACK( sum_dw_l, 1, dw_l.cols );
			{
				MAT_ON_STACK( ident_mat, 1, dw_l.rows );
				mat_fill( ident_mat, 1 );

				mat_dot( sum_dw_l, ident_mat, dw_l );
			}

			for (size_t _r = 0; _r < sum_dw_l.rows; _r++) 
				for (size_t _c = 0; _c < sum_dw_l.cols; _c++) 
					MAT_AT(sum_dw_l, _r, _c) = MAT_AT(sum_dw_l, _r, _c) / n;

			Mat avg_dw_l = sum_dw_l;

			for (size_t j=0; j < avg_dw_l.cols; j++) 
				MAT_AT( g->w[l], j, i ) = MAT_AT( avg_dw_l, 0, j );
		}	
	}
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


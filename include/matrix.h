#ifndef MAT_H
#define MAT_H

#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
	size_t  rows, cols;
	size_t stride; 
	float *p;
} Mat;

Mat mat_alloc(size_t row, size_t col);
Mat mat_dot(Mat out, Mat a, Mat b);
Mat mat_rand(Mat, float lower, float upper);
Mat mat_add(Mat out, Mat a, Mat b);
void mat_print(Mat, const char*, int);

Mat mat_sharrow(Mat, size_t);
Mat mat_sharcol(Mat, size_t);
Mat mat_cpy(Mat, Mat);

#ifndef MAT_ASSERT
	#define MAT_ASSERT assert
#endif

#ifndef MALLOC
	#define MALLOC malloc
#endif

#ifndef CALLOC
	#define CALLOC calloc
#endif

#ifndef RANDF
	#define RANDF ( (float) rand() / (float) RAND_MAX )
#endif

#define MAT_IMM_ALLOC( r, c ) ( (Mat){.rows = r, .cols = c, .p = (float[(r)*(c)]){0} )

#define MAT_AT(m, r, c) ( (m).p[ ((m).stride * (r)) + (c) ] )
#define MAT_PRINT(m) mat_print(m, #m, 0)

#endif

#ifdef MAT_IMPLEMENTATION

	Mat mat_alloc(size_t row, size_t col) {
		MAT_ASSERT(row > 0); MAT_ASSERT(col > 0);
		Mat m =  {.rows = row, .cols = col, .stride = col};
		MAT_ASSERT( (m.p = CALLOC(row*col, sizeof( *(m.p) ))) != NULL);
		return m;
	}

	Mat mat_rand(Mat m, float lowerlimit, float upperlimit) {
		MAT_ASSERT(m.p != NULL);
		for (size_t r = 0; r < m.rows; r++) {
			for (size_t c = 0; c < m.cols; c++) {
				MAT_AT(m, r, c) = lowerlimit + ( RANDF * ( upperlimit - lowerlimit ) );
			}
		}
		return m;
	}

	void mat_print(Mat m, const char* name, int padding) {
		MAT_ASSERT(name != NULL);
		printf("%*s%s: [\n", padding, "", name);
		for (size_t r = 0; r < m.rows; r++) {
			printf("%*s", padding, "");
			for (size_t c = 0; c < m.cols; c++) {
				printf("%s  %f ", "", MAT_AT(m, r, c));
			}
			printf("\n");
		}
		printf("%*s%s", padding, "",  "]\n");
	}

	Mat mat_dot(Mat out, Mat a, Mat b) {
		MAT_ASSERT(a.p != NULL); MAT_ASSERT(b.p != NULL); MAT_ASSERT(out.p != NULL);
		MAT_ASSERT(a.cols == b.rows); MAT_ASSERT( (a.rows == out.rows) && (b.cols == out.cols) );

		for (size_t c=0; c < b.cols; c++) {
			for (size_t n=0; n < a.rows; n++) {
				MAT_AT(out, n, c) = 0;
				for (size_t r=0; r < b.rows; r++) {
					MAT_AT(out, n, c) += MAT_AT(b, r, c) * MAT_AT(a, n, r);
				}
			}
		}

		return out;
	}

	Mat mat_add(Mat out, Mat a, Mat b) {
		MAT_ASSERT( (a.cols == b.cols) && (a.rows == b.rows) );
		MAT_ASSERT( (out.rows == a.rows) && (out.cols == a.cols) );

		for (size_t r=0; r < a.rows; r++) {
			for (size_t c=0; c < a.cols; c++) {
				MAT_AT(out, r, c) = MAT_AT(a, r, c) + MAT_AT(b, r, c);
			}
		}
		return out;
	}

	Mat mat_brcst(Mat out, Mat x, Mat b) {
		MAT_ASSERT(x.cols == b.cols);
		MAT_ASSERT(out.rows == x.rows);
		MAT_ASSERT(out.cols == x.cols);
		MAT_ASSERT(b.rows == 1);

			
		for (size_t i=0; i<x.rows; i++)  
			mat_add(mat_sharrow(out, i), b, mat_sharrow(x, i));

		return out;
	}

	Mat mat_sharrow(Mat m, size_t s) {
		Mat row = { .stride = m.stride, .rows = 1, .cols = m.cols, .p = m.p + (m.stride * s) };
		return row;	
	}

	Mat mat_sharcol(Mat m, size_t s) {
		Mat col  = { .stride = m.stride, .rows = m.rows, .cols = 1,  .p = m.p + s };  
		return col;
	}

	Mat mat_share(Mat src) {
		Mat dst = {
			.cols = src.cols, 
			.rows = src.rows, 
			.stride = src.stride, 
			.p = src.p,
		};
		return dst;
	}

	Mat mat_cpy(Mat dst, Mat src) {
		MAT_ASSERT(dst.rows >= src.rows);
		MAT_ASSERT(dst.cols >= src.cols);
		
		dst.rows = src.rows;
		dst.cols = src.cols;

		for (size_t i = 0; i < src.rows; i++) 
			for (size_t j = 0; j < src.cols; j++) 
				MAT_AT(dst, i, j) = MAT_AT(src, i, j);
		
		return dst;
	}



#endif







#ifndef MAT_H
#define MAT_H

#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
	size_t  rows, cols;
	float *p;
} Mat;

Mat Mat_init(size_t row, size_t col);
Mat Mat_dot(Mat out, Mat a, Mat b);
Mat Mat_rand(Mat, float lower, float upper);
Mat Mat_add(Mat out, Mat a, Mat b);
void Mat_print(Mat, const char*);

#ifndef ASSERT
	#define ASSERT assert
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

#define MAT_AT(m, r, c) ( m.p[ (m.cols * r) + c ] )
#define MAT_PRINT(m) Mat_print(m, #m)

	#ifdef MAT_IMPLEMENTATION

	Mat Mat_init(size_t row, size_t col) {
		ASSERT(row > 0); ASSERT(col > 0);
		Mat m =  {.rows = row, .cols = col};
		ASSERT( (m.p = CALLOC(row*col, sizeof( *(m.p) ))) != NULL);
		return m;
	}

	Mat Mat_rand(Mat m, float lowerlimit, float upperlimit) {
		ASSERT(m.p != NULL);
		for (size_t r = 0; r < m.rows; r++) {
			for (size_t c = 0; c < m.cols; c++) {
				MAT_AT(m, r, c) = lowerlimit + ( RANDF * ( upperlimit - lowerlimit ) );
			}
		}
		return m;
	}

	void Mat_print(Mat m, const char* name) {
		ASSERT(name != NULL);
		printf("%s: [\n", name);
		for (size_t r = 0; r < m.rows; r++) {
			for (size_t c = 0; c < m.cols; c++) {
				printf("%f\t", MAT_AT(m, r, c));
			}
			printf("\n");
		}
		printf("]\n");
	}

	Mat Mat_dot(Mat out, Mat a, Mat b) {
		ASSERT(a.p != NULL); ASSERT(b.p != NULL); ASSERT(out.p != NULL);
		ASSERT(a.cols == b.rows); ASSERT( (a.rows == out.rows) && (b.cols == out.cols) );

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

	Mat Mat_add(Mat out, Mat a, Mat b) {
		ASSERT( (a.cols == b.cols) && (a.rows == b.rows) );
		ASSERT( (out.rows == a.rows) && (out.cols == a.cols) );

		for (size_t r=0; r < a.rows; r++) {
			for (size_t c=0; c < a.cols; c++) {
				MAT_AT(out, r, c) = MAT_AT(a, r, c) + MAT_AT(b, r, c);
			}
		}
		return out;
	}

	#endif

#endif







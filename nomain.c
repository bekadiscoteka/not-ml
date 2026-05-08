#define MAT_IMPLEMENTATION
#include <matrix.h>
#include <stdio.h>



int main(void) {
	Mat a = {.rows = 2, .cols = 2, .p = (float[]) {1.0F, 4.0F, 3.0f, 3.0f}};
	Mat b = {.rows = 2, .cols = 1, .p = (float[]) {2.0F, 3.0F}};
	Mat c = Mat_init(2, 1);
	Mat d = Mat_init(2, 1);	
	Mat out = Mat_init(2, 1);
	Mat_rand(d, 0, 1);
	
	MAT_PRINT(a); MAT_PRINT(b);

	Mat_dot(c, a, b);

	MAT_PRINT(c);

	Mat_add(out, (Mat){2, 1, (float[]) {2.0f, 2.0f}}, c);
	
	MAT_PRINT(out);
	return 0;
}

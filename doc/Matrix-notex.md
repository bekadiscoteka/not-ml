## `Mat` (matrix.h)
 
```c
typedef struct {
    size_t rows, cols, stride;
    float *p;
} Mat;
```
 
`stride` lets a `Mat` be a *view* into a larger buffer (row/col/submatrix slices) without copying data. Indexing always goes through `stride`, never `cols`.
 
### Macros
 
| Macro | Description |
|---|---|
| `MAT_AT(m, r, c)` | Element access, `m.p[m.stride*r + c]` |
| `MAT_ON_STACK(name, r, c)` | Declares a stack-allocated `Mat name` |
| `MAT_PRINT(m)` | `mat_print(m, "m", 0)` |
| `MAT_FOREACH(m, op, f)` | Applies `m[i][j] op= f` in place |
 
### Allocation & init
 
| Function | Description |
|---|---|
| `mat_alloc(rows, cols)` | Heap-allocate a zeroed matrix |
| `mat_rand(m, lo, hi)` | Fill with uniform randoms in `[lo, hi)` |
| `mat_fill(m, v)` | Fill with constant `v` |
| `mat_cpy(dst, src)` | Copy values, resizes `dst`'s logical `rows`/`cols` to `src`'s |
 
### Views (no copy — share underlying `p`)
 
| Function | Description |
|---|---|
| `mat_sharrow(m, i)` | Row `i` as a `1 x cols` view |
| `mat_sharcol(m, i)` | Column `i` as a `rows x 1` view |
| `mat_sharsub(m, ri, rsize, ci, csize)` | Rectangular sub-view |
 
### Arithmetic
 
| Function | Description |
|---|---|
| `mat_dot(out, a, b)` | Matrix product, `out = a @ b` |
| `mat_add(out, a, b)` | Elementwise add |
| `mat_subtr(out, a, b)` | Elementwise subtract, `out = a - b` |
| `mat_mul(out, a, b)` | Elementwise (Hadamard) product |
| `mat_transpose(out, m)` | `out = m^T` |
| `mat_brcst(out, x, b)` | Add row vector `b` (`1 x cols`) to every row of `x` |
| `mat_colmean(out, m)` | Column-wise mean, `out` is `1 x cols` |

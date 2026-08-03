
A tiny, single-header neural network framework in C. 
- no dependencies, 
- no layer types (conv, attention, etc.)
- no serialization
- no GPU acceleration 
- no opinions,  
- no job, 
- no money, 
- no taxes, 
- no corruption, 
- no grass touch, 
- no money again
- no pork,
- no ad,
- no trust,
- no shame,
- no mood,
- no ai,
- no belief in future,
- no cries
- no motivation
- no html,css
- nocopyright 
- no responsibility 
- no arch linux
- no one has been harmed while writing this code
- no smoking
 —
# not-ml


```
include/
  matrix.h   -> Mat type + linear algebra
  NN.h       -> NN type + forward/backward/train (depends on matrix.h)
```
 
## Usage
 
Define the implementation macros in exactly one `.c` file before including:
 
```c
#define MAT_IMPLEMENTATION
#define NN_IMPLEMENTATION
#include <NN.h>
```
 
Every other translation unit just includes the headers normally (declarations only).
 
## Quick start
 
```c
size_t arch[] = {2, 2, 1};                    // 2 inputs, 2 hidden, 1 output
NN nn = nn_alloc(arch, 3, batch);
nn_rand(&nn);
 
mat_cpy(NN_INPUT(&nn), x);                    // x: (batch, 2)
nn_forward(&nn);
 
NN g = nn_alloc(arch, 3, batch);              // gradient buffer, same shape
for (size_t epoch = 0; epoch < 100000; epoch++) {
    mat_cpy(NN_INPUT(&nn), x);
    nn_forward(&nn);
    nn_backward(&nn, &g, y);                  // y: (batch, 1)
    nn_train(&nn, &g, 0.01f);
}
```
 


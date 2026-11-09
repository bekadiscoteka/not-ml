# not-ml

A small, dependency-free neural network library in C, written from scratch to understand how training actually works: matrices, forward pass, backpropagation and gradient descent, with nothing hidden behind a framework.

It is header-only: two headers, no build system required beyond a C compiler and `libm`.

```
include/
  matrix.h   Mat type and linear algebra
  NN.h       NN type: allocation, forward pass, backpropagation, training (depends on matrix.h)
test/
  xor.c          learns XOR with a 2-2-1 network
  full_adder.c   learns a 1-bit full adder (sum and carry)
doc/
  Matrix-notex.md, NN-notes.md   notes on the data structures and the maths
```

## Features

- Fully connected networks of any depth, defined by an array of layer sizes
- Batched forward pass and backpropagation over a whole dataset at once
- Gradient-descent training with a configurable learning rate
- Matrix views (`mat_sharsub`, `mat_sharcol`, `mat_sharrow`) that slice a dataset into inputs and labels without copying
- Pluggable allocator and assert (`NN_MALLOC`, `NN_CALLOC`, `NN_ASSERT`)

Current limits: sigmoid activation only, dense layers only, no model serialization, CPU only.

## Usage

Define the implementation macros in exactly one `.c` file before including the header:

```c
#define MAT_IMPLEMENTATION
#define NN_IMPLEMENTATION
#include <NN.h>
```

Every other translation unit includes the header normally and gets declarations only.

## Quick start

```c
size_t arch[] = {2, 2, 1};                    // 2 inputs, 2 hidden, 1 output
NN nn = nn_alloc(arch, 3, batch);
nn_rand(&nn);

NN g = nn_alloc(arch, 3, batch);              // gradient buffer, same shape
for (size_t epoch = 0; epoch < 100000; epoch++) {
    mat_cpy(NN_INPUT(&nn), x);                // x: (batch, 2)
    nn_forward(&nn);
    nn_backward(&nn, &g, y);                  // y: (batch, 1)
    nn_train(&nn, &g, 0.01f);
}

MAT_PRINT(NN_OUTPUT(&nn));
```

## Build and run the examples

```bash
make        # builds every file in test/ into bin/
make run    # builds and runs them
make clean
```

## Why

Most ML code starts at the framework level. This project starts one level lower, so every gradient is code you can read and step through in a debugger. The notes in `doc/` explain the shapes and the derivation behind `nn_backward`.

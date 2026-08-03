## `NN` (NN.h)
 
```c
typedef struct {
    Mat *w, *b, *z, *a;   // one entry per layer, index 0 = input layer
    size_t size;          // number of layers, including input
} NN;
```
 
- `a[i]` — activations of layer `i` (`a[0]` is the input, set manually)
- `z[i]` — pre-activation (`w·a + b`) of layer `i`
- `w[i]`, `b[i]` — weights/bias into layer `i` (unused for `i == 0`)
- Activation function: sigmoid, hardcoded.
### Macros
 
| Macro | Description |
|---|---|
| `NN_INPUT(nn)` | `nn->a[0]` |
| `NN_OUTPUT(nn)` | `nn->a[nn->size - 1]` |
| `NN_PRINT(nn)` | `nn_print(nn, "nn")` |
 
### Functions
 
| Function | Description |
|---|---|
| `nn_alloc(arch, size, batch)` | Allocates all layers. `arch[i]` = neuron count of layer `i`; `size` = layer count; `batch` = fixed sample count per forward/backward pass |
| `nn_rand(nn)` | Randomize all `w`/`b` in `[0, 1)` |
| `nn_forward(nn)` | Propagates `a[0]` → `a[size-1]` through `z = a·w + b`, `a = sigmoid(z)` |
| `nn_backward(nn, g, y)` | Backprop against target `y` (MSE loss); writes weight/bias gradients into `g` (an `NN` of identical shape) |
| `nn_train(nn, g, lr)` | `w -= lr*g.w`, `b -= lr*g.b` |
| `nn_print(nn, name)` | Dumps weights and biases |
 
### Training loop shape
 
Batch size is fixed at `nn_alloc` time (full-batch only — no minibatching). A typical epoch:
 
```c
mat_cpy(NN_INPUT(&nn), x);
nn_forward(&nn);
nn_backward(&nn, &g, y);
nn_train(&nn, &g, lr);
```
 
`nn` and `g` must be allocated with the same `arch`/`batch`; `g` only ever holds gradients, never randomized.

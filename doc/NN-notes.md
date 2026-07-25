# little documentation, further to be improved
**input matrix works only with single row** for now
Each layer is represented by following matrices:
- weight matrix:
	- each *column* represents a neuron
	- each *row* represents weights of that neuron
- bias matrix:
	- each *column* is each neuron's bias
	- each neuron has one bias value as a row
- activation matrix:
	- each *column* is each neuron's activation
	
NN struct: 
- following var-s represent **array of**:
	w - *weight* matrices 
	b - *bias* matrices 
	a - *activation* matrices
	**for each layer**
- size = size of those matrices


```
size_t arch[] = {3, 2, 1};
nn_alloc(arch, sizeof(arch));
```


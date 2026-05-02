#include <stdio.h>
#include <math.h>
#include <NN.h>

NN *nn_init(NN *nn_inst, unsigned input_size, unsigned init_set[], size_t size, float (*actFunc) (float)) {
	nn_inst->INPUT_SIZE = input_size;
	nn_inst->LAYER_N = size;
	for (int l=0; l<size; l++) {
		Layer l_inst;
		l_inst.NEURON_N = init_set[l];	
		for (int n=0; n<l_inst.NEURON_N; n++) {
			Neuron n_inst;
		   	n_inst.WEIGHT_N = input_size;
			for (int w=0; w<n_inst.WEIGHT_N; w++) {
				n_inst.weights[w] = RANDF;
				
			}	
			n_inst.bias = RANDF;
			l_inst.neurons[n] = n_inst;
		}
		nn_inst->layers[l] = l_inst;
		input_size = l_inst.NEURON_N;
	}	
	nn_inst->actFunc = actFunc;
	return nn_inst;
}

void farrncpy(float t[], float s[], size_t n) {
	for (size_t i=0; i<n; ++i) 
		t[i] = s[i];
}

float forward(const NN *nn, float input[MAXWEIGHT_N]) {
	float next_layer_input[MAXWEIGHT_N];
	for (int i=0; i<nn->LAYER_N; ++i) {
		for (int j=0; nn->layers[i].NEURON_N; j++) {
			float result = 0;
			for (int k=0; k < nn->layers[i].neuron[j].WEIGHT_N; ++k) {
				result += nn->layers[i].neuron[j].weights[k] * input[k];
			}
			result += nn->layers[i].neuron[j].bias;
			next_layer_input[j] = (*(nn->actFunc))(result);
		}
		farrcpy(input, next_layer_input, MAXWEIGHT_N);
	}
		
}

float MSE(const NN *nn, const float *dataset[], size_t datasize) {
	float mse = 0;
	for (size_t i=0; i<datasize; ++i) {
		float err = dataset[i][TARGET] - forward(nn, dataset[i]);
		mse += err * err;
	}
	return mse;
}	


void train(float dataset[], size_t datasize, const NN *orig, float eps, float lr, int epoch) {
	NN step, clone;
	clone = step = *orig;

	for (int e=0; e < epoch; ++e) { 
		for (int i=0; i < orig->LAYER_N; i++) {
			for (int j=0; j < orig->layers[i].NEURON_N; j++) {
				for (int k=0; k < orig->layers[i].neurons[j].WEIGHT_N; k++) {
					step.layers[i].neurons[j].weights[k] += eps;
					float diff = MSE(orig, dataset, datasize) - MSE(step, dataset, datasize); 
					step.layers[i].neurons[j].weight[k] -= eps;

					clone.layers[i].neurons[j].weight[k] -= lr * (diff / eps);
				}

				step.layers[i].neurons[j].bias += eps;
				float diff = MSE(orig, dataset, datasize) - MSE(step, dataset, datasize); 
				step.layers[i].neurons[j].bias -= eps;
				clone.layers[i].neurons[j].bias -= lr * (diff / eps);
			}
		}	
	}

}	



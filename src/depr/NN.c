#include <stdio.h>
#include <math.h>
#include <NN.h>
#include <stdlib.h>

#define RANDF ( (float) ( rand() / ((float) RAND_MAX) ) ) 

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

void printmodel(FILE *output, NN *model) {
	fprintf(output, "INPUT SIZE: %d, LAYERS: %d\n", model->INPUT_SIZE, model->LAYER_N);
	for (int i=0; i<model->LAYER_N; i++) {
		Layer *l = model->layers + i;
		fprintf(output, "\tLAYER-#%d, NEURONS: %d\n", i, l->NEURON_N);
		for (int j=0; j<l->NEURON_N; j++) {
			Neuron *n = l->neurons + j;
			fprintf(output, "\t\tNeuron-#%d, WEIGHTS: %d\n", j, n->WEIGHT_N);
			for (int k=0; k<n->WEIGHT_N; k++) {
				fprintf(output, "\t\t\tWeight-#%d = %f\n", k, n->weights[k]);
			}
		}
	}

}

void farrncpy(float t[], float s[], size_t n) {
	for (size_t i=0; i<n; ++i) 
		t[i] = s[i];
}


float *forward(float output[], const NN *nn, float input[], size_t inputsize) {
	float next_layer_input[MAXNEURON_N], current_layer_input[MAXNEURON_N];
	farrncpy(current_layer_input, input, inputsize);
	for (int i=0; i<nn->LAYER_N; i++) {
		Layer *l = nn->layers + i;
		for (int j=0; j<l->NEURON_N; j++) {
			Neuron *n = l->neurons + j;
			float result = 0;

			for (int k=0; k<n->WEIGHT_N; k++) 
				result += n->weights[k] * current_layer_input[k];	

			result += n->bias;
			next_layer_input[j] = (*nn->actFunc)(result);
		}
		farrncpy(current_layer_input, next_layer_input, l->NEURON_N);
	}
	farrncpy(output, current_layer_input, (nn->layers[nn->LAYER_N - 1]).NEURON_N);
	return output; 
}

float MSE(const NN *nn, Dataset *ds) {
	float mse = 0;
	size_t TARGET = ds->target;
	for (size_t i=0; i<ds->datasize; ++i) {
		float output[3];
		float err = ds->dataset[(i * (ds->feature_size + 1)) + TARGET]	- (forward(output, nn, ds->dataset + (i * (ds->feature_size + 1)), 2))[0];
		mse += err * err;
	}
	return mse;
}	

	

void train(Dataset *ds, NN *orig, float eps, float lr, unsigned int epoch) {
	NN step, clone;
	step = clone = *orig;
	for (unsigned int e=0; e < epoch; ++e) {
			
		for (int i=0; i < orig->LAYER_N; i++) {
			Layer *l = orig->layers + i;
					
			for (int j=0; j < l->NEURON_N; j++) {
				Neuron *n = l->neurons + j;
				
				for (int k=0; k < n->WEIGHT_N; k++) {
					float mse_orig = MSE(orig, ds);
					n->weights[k] += eps;
					float mse_eps = MSE(orig, ds);
					n->weights[k] -= eps;
					float grad = ( mse_orig - mse_eps ) / eps;
					clone.layers[i].neurons[j].weights[k] += lr * grad;
				}
				
				float mse_orig = MSE(orig, ds);
				n->bias += eps;
				float mse_eps = MSE(orig, ds);
				n->bias -= eps;
				float grad = ( mse_orig - mse_eps ) / eps;
				clone.layers[i].neurons[j].bias += lr * grad;
			}
		}
		//printf("epoch = %d, MSE: %f\n", e, MSE(orig, ds));
		*orig = clone;
	}
}

float SIGMOID(float x) {
	return (1.0F / ( 1 + exp(-x) ));
}






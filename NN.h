#include <stddef.h>
#include <stdio.h>
#define MAXWEIGHT_N 5
#define MAXNEURON_N 5
#define MAXLAYER_N 5

typedef struct {
	unsigned int WEIGHT_N;
	float weights[MAXWEIGHT_N];
	float bias;
} Neuron;

typedef struct {
	unsigned int NEURON_N;
	Neuron neurons[MAXNEURON_N];
} Layer;

typedef struct {
	unsigned int INPUT_SIZE, LAYER_N;
	Layer layers[MAXLAYER_N];
	float (*actFunc)(float);
} NN;


void printmodel(FILE *output, NN *model);
float SIGMOID(float); 
NN *nn_init(NN *nn, unsigned input_size, unsigned init_set[], size_t size, float (*actFunc) (float));
void train(float dataset[][3], size_t datasize, NN *orig, float eps, float lr, unsigned int epoch);
float MSE(const NN *nn, const float dataset[][3], size_t datasize);
float *forward(float output[], const NN *nn, float input[], size_t size);
void farrncpy(float t[], float s[], size_t n);

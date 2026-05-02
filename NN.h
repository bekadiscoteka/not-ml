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
	Layer layers[MAXLAYER];
	float (*actFunc)(float);
} NN;


NN *nn_init(NN *nn, unsigned input_size, unsigned init_set[], size_t size, float (*actFunc) (float));
float train(float dataset[], size_t datasize, const NN *orig, float eps, float lr, int epoch);
float MSE(const NN *nn, const float *dataset[], size_t datasize);
float forward(const NN *nn, float input[MAXWEIGHT_N]);

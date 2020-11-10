#include "cnn_inference.h"
#include "h5_format.h"
#include "rgb_format.h"
#include <time.h>

int main(int argc, char *argv[]){


	clock_t start, end;
	double cpu_time_used;

	//Prepare Input
	float ***img;
	img = load_RGB(argv[1], 1, 24,129);

	Tensor *t;
	t = make_tensor(1, 24, 129, img);

	//Generate Layers
	ConvLayer *_conv2d_1;
	_conv2d_1 = empty_Conv(32, 1, 3, 4, 3, 2, SAME);
	load_Conv(_conv2d_1, "conv2d_1");

	ConvLayer *_conv2d_2;
	_conv2d_2 = empty_Conv(48, 32, 1, 6, 1, 3, SAME);
	load_Conv(_conv2d_2, "conv2d_2");

	ConvLayer *_conv2d_3;
	_conv2d_3 = empty_Conv(64, 48, 2, 4, 1, 3, SAME);
	load_Conv(_conv2d_3, "conv2d_3");

	DenseLayer *_dense_1;
	_dense_1 = empty_Dense(64, 512, 1, 1);
	load_Dense(_dense_1, "dense_1");

	DenseLayer *_dense_2;
	_dense_2 = empty_Dense(1, 64, 1, 1);
	load_Dense(_dense_2, "dense_2");

	//Inference
	start = clock();

	t = Conv(t, _conv2d_1, ReLU_activation, 1);
	t = MaxPool(t, 1, 3, 1, 2, SAME, 1);
	t = Conv(t, _conv2d_2, ReLU_activation, 1);
	t = MaxPool(t, 1, 3, 1, 2, SAME, 1);
	t = Conv(t, _conv2d_3, ReLU_activation, 1);
	t = MaxPool(t, 1, 3, 1, 2, SAME, 1);
	t = FlattenD(t, 1);
	t = Dense(t, _dense_1, ReLU_activation, 1);
	t = Dense(t, _dense_2, linear_activation, 1);

	end = clock();

	print_tensor(t);

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\nInference completed in: %f", cpu_time_used);

	//Free Memory
	free_ConvLayer(_conv2d_1);
	free_ConvLayer(_conv2d_2);
	free_ConvLayer(_conv2d_3);
	free_DenseLayer(_dense_1);
	free_DenseLayer(_dense_2);
}
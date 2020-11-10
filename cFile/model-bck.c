#include "cnn_inference.h"
#include "h5_format.h"
#include "rgb_format.h"
#include <time.h>
#include <stdio.h>
#include < stdlib.h>

void show_matrix(Tensor *t, int limit_d, int limit_h, int limit_w) {
	FILE *fp = NULL;
	char *ptr;
	int decpt, sign;

	/*fp = fopen("record.txt", "w+");
	if (fp == NULL || fp == NULL) {
		printf("open file failed.\n");
		return;
	}

	for (int i = 0; i < limit_d; i++)
	{
		for (int j = 0; j < limit_h; j++)
		{
			for (int k = 0; k < limit_w; k++)
			{
				fprintf(fp,"%f", t->T[i][j][k]);
				fprintf(fp,"%s"," ");
			}
			fprintf(fp, "%s", "\n");
		}
		fprintf(fp,"%s","\n\n");
	}
	fclose(fp);*/

	for (int i = 0; i < limit_d; i++)
	{
		for (int j = 0; j < limit_h; j++)
		{
			for (int k = 0; k < limit_w; k++)
			{
				printf("%f", t->T[i][j][k]);
				printf(" ");
			}
			printf("\n");
		}
		printf("\n\n");
	}
}

int main(int argc, char *argv[]){

	/*if(argc!=2){
		printf("\nPlease run as: model.c <image_name>. Image name must be given without the 0,1,2 and without its extension.\n");
		exit(EXIT_FAILURE);
	}*/

	clock_t start, end;
	double cpu_time_used;

	//Prepare Input
	float ***img;
	/*img = load_RGB(argv[1], 28, 28);*/
	img = load_RGB("pic",1, 28, 28);

	Tensor *t;
	t = make_tensor(1, 28, 28, img);
	//show_matrix(t, 1, 28, 28);

	//Generate Layers
	ConvLayer *_conv2d_1;
	_conv2d_1 = empty_Conv(32, 1, 5, 5, 3, 3, SAME);
	load_Conv(_conv2d_1, "conv2d_1");

	DenseLayer *_dense_1;
	_dense_1 = empty_Dense(128, 800, 1, 1);
	load_Dense(_dense_1, "dense_1");

	DenseLayer *_dense_2;
	_dense_2 = empty_Dense(10, 128, 1, 1);
	load_Dense(_dense_2, "dense_2");

	//Inference
	start = clock();

	t = Conv(t, _conv2d_1, ReLU_activation, 1);
	//show_matrix(t, 32, 10, 10);
	t = MaxPool(t, 2, 2, 2, 2, VALID, 1);
	//show_matrix(t, 32, 5, 5);
	t = FlattenD(t, 1);
	//show_matrix(t, 10, 1, 1);
	t = Dense(t, _dense_1, ReLU_activation, 1);
	show_matrix(t, 10, 1, 1);
	t = Dense(t, _dense_2, softmax_activation, 1);
	show_matrix(t, 10, 1, 1);

	end = clock();

	print_tensor(t);

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\nInference completed in: %f", cpu_time_used);

	//Free Memory
	free_ConvLayer(_conv2d_1);
	free_DenseLayer(_dense_1);
	free_DenseLayer(_dense_2);
}
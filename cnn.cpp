#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "weights.h"

#define INSIZE 28

typedef struct mnist_data{
	double data[INSIZE][INSIZE];
	unsigned int label;
}mnist_data;


const char *train_label_filename="data/train-labels.idx1-ubyte";
const char *train_image_filename="data/train-images.idx3-ubyte";

const char *test_label_filename="data/t10k-labels.idx1-ubyte";
const char *test_image_filename="data/t10k-images.idx3-ubyte";

//https://bytes.com/topic/c/answers/577180-converting-big-endian-little-endian
static unsigned int swap_endian(unsigned int b){
	return (b>>24) | ((b>>8) & 0x0000ff00) | ((b<<8) & 0x00ff0000) | (b<<24);
}

static unsigned int mnist_bin_to_int(char* tmp){
	return (*tmp) & 0x000000ff;
}

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count);

__global__ void kernel_convolution(double data[28][28], float weight[6][5][5], float bias[6], float output[6][24][24], int IN=28, int N=6, int M=5){
	int Oy = IN-M+1;  //height
	int Oz = IN-M+1;  //width
	int l = threadIdx.x + blockIdx.x * blockDim.x; //output layer depth, x-axis index
	int i = threadIdx.y + blockIdx.y * blockDim.y; // output y-axis index,
	int j = threadIdx.z + blockIdx.z * blockDim.z; // output z-axis index
	if(l >= N or i >= Oy or j >= Oz){
	    return;
	}
	int ii,jj;
	float tmp=0.0;

	//filtering
	for(ii=0; ii<M; ii++){
	    for(jj=0; jj<M; jj++){
	        tmp += weight[l][ii][jj] * data[ii+i][jj+j];     
	    }
	}

	//bias
	tmp += bias[l];

	//activation
	output[l][i][j]=1.0/(1.0+exp(-tmp));
}
__global__ void kernel_subsampling(float data[6][24][24], float output[6][6][6], float weight[1][4][4], float bias[1]){
    int ii,jj;
  
    int l = threadIdx.x + blockIdx.x * blockDim.x; //output layer depth, x-axis index
    int i = threadIdx.y + blockIdx.y * blockDim.y; // output y-axis index,
    int j = threadIdx.z + blockIdx.z * blockDim.z; // output z-axis index
    if(l >= 6 or i >= 24 or j >= 24 or i%4!=0 or j%4!=0){
        return;
    }
    //matrix multiplication
    float tmp=0;
    for(ii=0;ii<4;ii++){
        for(jj=0;jj<4;jj++){
            tmp += data[l][i+ii][j+jj] * weight[0][ii][jj];
        }
    }
    //bias
    tmp+=bias[0];
    //sigmoid
    output[l][i/4][j/4] = 1.0/(1.0+exp(-tmp));
}
__global__ void kernel_fully_connected(float data[6][6][6], float weight[10][6][6][6], float bias[10], float output[10]){

    int l = threadIdx.x + blockIdx.x * blockDim.x; //output layer depth, x-axis index

    if(l >= 10){
      return;
    }
    int i,j,k;
    float tmp=0;
    for(i=0;i<6;i++){
      for(j=0;j<6;j++){
        for(k=0;k<6;k++){
          tmp += weight[l][i][j][k] * data[i][j][k];
        }
      }
    }
    tmp += bias[l];
    tmp = 1.0/(1.0+exp(-tmp));
    output[l]=tmp;
}

class CNN{
    public:

    int M,N,O;
    float weight1[6][5][5], bias1[6];
    float weight2[1][4][4], bias2[1];
    float weight3[10][6][6][6], bias3[10];

    float output[10];

    CNN(){
    	//initialize with given trainable parameter values; 
    	for(int i=0;i<6;i++){
    		bias1[i]=c1_bias[i];
    		for(int j=0;j<5;j++){
    			for(int k=0;k<5;k++){
    				weight1[i][j][k]=c1_weight[i][j*5+k];
    			}
    		}
    	}

    	for(int i=0;i<1;i++){
    		bias2[i]=s2_bias[i];
    		for(int j=0;j<4;j++){
    			for(int k=0;k<4;k++){
    				weight2[i][j][k]=s2_weight[i][j*4+k];
    			}
    		}
    	}

    	for(int l=0;l<10;l++){
            bias3[l]=f3_bias[l];
            for(int i=0;i<6;i++){
                for(int j=0;j<6;j++){
                    for(int k=0;k<6;k++){
                        weight3[l][i][j][k]=f3_weight[l][i*6*6 + j*6 + k];
                    }
                }
            }
        }

    }
    ~CNN(){
    }

    float forward_pass(double  data[28][28]){

/* ----------------CONVOLUTION-----------------*/
        float (*dWeight1)[5][5],(*dOutput1)[24][24], *dBias1;
        double (*dData)[28];

        float ms,total=0;

        cudaMalloc(&dData,sizeof(double)*28*28);
        cudaMalloc(&dWeight1,sizeof(float)*6*5*5);
        cudaMalloc(&dOutput1,sizeof(float)*6*24*24);
        cudaMalloc(&dBias1,sizeof(float)*6);

        cudaMemcpy(dData,data,sizeof(double)*28*28, cudaMemcpyHostToDevice);
        cudaMemcpy(dWeight1,weight1,sizeof(float)*6*5*5, cudaMemcpyHostToDevice);
        cudaMemcpy(dBias1,bias1,sizeof(float)*6, cudaMemcpyHostToDevice);

        dim3 blocks1(2,4,8),threads1(3,6,3);

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        kernel_convolution<<<blocks1,threads1>>>(dData, dWeight1, dBias1, dOutput1,28,6,5);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        cudaError_t cudaerror=cudaGetLastError();
        if ( cudaerror!= cudaSuccess ){
            printf("Could not run kernel: %s\\n",cudaGetErrorString(cudaerror));
        }

        cudaFree(dData);
        cudaFree(dWeight1);
        cudaFree(dBias1);
        total += ms;

/* ----------------Sub sampling-----------------*/

        float (*dWeight2)[4][4], (*dOutput2)[6][6], *dBias2;
        cudaMalloc(&dWeight2,sizeof(float) * 1*4*4);
        cudaMalloc(&dOutput2,sizeof(float) * 6*6*6);
        cudaMalloc(&dBias2,sizeof(float) * 1);

        cudaMemcpy(dWeight2,weight2, sizeof(float) * 1*4*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dBias2,bias2, sizeof(float) * 1, cudaMemcpyHostToDevice);
        
        dim3 blocks2(2,4,4);
        dim3 threads2(3,6,6);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        kernel_subsampling<<<blocks2,threads2>>>(dOutput1,dOutput2,dWeight2,dBias2);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        cudaerror=cudaGetLastError();
        if ( cudaerror!= cudaSuccess ){
            printf("Could not run kernel: %s\\n",cudaGetErrorString(cudaerror));
        }
        cudaFree(dOutput1);
        cudaFree(dWeight2);
        cudaFree(dBias2);
        total += ms;


/* ----------------Fully connected-----------------*/

        float (*dWeight3)[6][6][6], *dOutput3, *dBias3;
        cudaMalloc(&dWeight3,sizeof(float) * 10*6*6*6);
        cudaMalloc(&dOutput3,sizeof(float) * 10);
        cudaMalloc(&dBias3,sizeof(float) * 10);

        cudaMemcpy(dWeight3,weight3, sizeof(float) * 10*6*6*6, cudaMemcpyHostToDevice);
        cudaMemcpy(dBias3,bias3, sizeof(float) * 10, cudaMemcpyHostToDevice);
        
        dim3 blocks3(2,1,1);
        dim3 threads3(5,1,1);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        kernel_fully_connected<<<blocks3,threads3>>>(dOutput2,dWeight3,dBias3,dOutput3);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        cudaerror=cudaGetLastError();
        if ( cudaerror!= cudaSuccess ){
            printf("Could not run kernel: %s\\n",cudaGetErrorString(cudaerror));
        }

        cudaMemcpy(output,dOutput3,sizeof(float) * 10, cudaMemcpyDeviceToHost);
        
        cudaFree(dOutput2);
        cudaFree(dWeight3);
        cudaFree(dBias3);
        cudaFree(dOutput3);
        total += ms;

        return total;
    }
};


int main(){
	CNN cnn;
	mnist_data* data_set;
	unsigned int count=0;
	int read=mnist_load(test_image_filename,test_label_filename,&data_set,&count);
	
	if(read == 0){
		printf("test_cnt = %d (should be 10000)\\n\\n",count);
	}
	unsigned int error = 0;
	unsigned int max = 0;
	float time_taken=0;
	for(int i=0;i<count;i++){
		time_taken += cnn.forward_pass(data_set[i].data);

		for(int j=0;j<10;j++){
			if (cnn.output[max] < cnn.output[j]){
				max = j;
			}
		}
		if (max != data_set[i].label) ++error;

	}
	printf("Error Rate = %f%% (%d out of 10,000)\\n", double(error)/double(count)*100.0, error);
	printf("Accuracy = %.3f%% (%d out of 10,000)\\n",100.0 - double(error)/double(count)*100.0, count - error);
	printf("Ex time = %f (ms) \\n", time_taken);

	free(data_set);
	return 0;
}

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count){
	FILE *image_file = fopen(image_filename,"rb");
	FILE *label_file = fopen(label_filename,"rb");

	unsigned int image_magic, image_number, image_row, image_column;
	unsigned int label_magic, label_number;
	
	fread((void*)&image_magic,4,1,image_file);
	fread((void*)&image_number,4,1,image_file);
	fread((void*)&image_row,4,1,image_file);
	fread((void*)&image_column,4,1,image_file);
	
	image_magic =  swap_endian(image_magic);
	image_number =  swap_endian(image_number);
	image_row =  swap_endian(image_row);
	image_column =  swap_endian(image_column);

	fread((void*)&label_magic,4,1,label_file);
	fread((void*)&label_number,4,1,label_file);
	
	label_magic =  swap_endian(label_magic);
	label_number =  swap_endian(label_number);


	printf("image magic number = %d (should be 2051)\\n",image_magic);
	printf("label magic number = %d (should be 2049)\\n",label_magic);
	printf("image total number = %d (should be 10000)\\n",image_number);
	printf("label total number = %d (should be 10000)\\n",label_number);
	printf("rows = %d cols = %d (both should be 28)\\n",image_row,image_column);
	
	(*data_set) = (mnist_data*)malloc(sizeof(mnist_data) * image_number);

	char image[INSIZE*INSIZE];
	char *labels = (char*)malloc(image_number * sizeof(char));
	
	if(image_number != fread((void*)labels,1,image_number,label_file)){
		printf("Error while reading labels");
	}
	
	*count = 0;
	int read_size;
	for(int k=0;k<image_number;k++){
		
		read_size = fread((void*)&image,1,INSIZE*INSIZE,image_file);
		if(read_size == INSIZE*INSIZE)
			(*count)++;
		else{
			printf("Error while reading image");
			break;
		}
		for(int i=0;i<image_row;i++){
			for(int j=0;j<image_column;j++){
				(*data_set)[k].data[i][j]=((double)mnist_bin_to_int(&image[i*INSIZE+j]))/255;
				(*data_set)[k].label = mnist_bin_to_int(&labels[k]);
			}
		}
	}
	free(labels);
	fclose(image_file);
	fclose(label_file);
	return 0;
}

/*START_CITE
https://bytes.com/topic/c/answers/577180-converting-big-endian-little-endian
Referred code 1
static unsigned int swap_endian(unsigned int b){
	return (b>>24) | ((b>>8) & 0x0000ff00) | ((b<<8) & 0x00ff0000) | (b<<24);
}
END_CITE*/



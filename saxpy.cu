#include <stdio.h>

__global__

void saxpy(int n,float a, float *x,float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) y[i] = a*x[i]+y[i];
}

int main(void)
{
    int N =8<<20;
    float *x, *y,*d_x,*d_y;
    float r,s,t,e;

    printf("\nEnter the 3 val=");
    scanf("%f%f%f",&r,&s,&t);
    printf("\nEnter the error correction val=");
    scanf("%f",&e);

    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    for(int i =0;i<N;i++){
        x[i] = s;
        y[i]= t;
    }

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    //Perform Saxpy on 1M elements
    saxpy<<<(N+255)/256,256>>>(N, r, d_x,d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyHostToDevice);

    float maxError = 0.0f;
    for(int i=0;i<N;i++)
        maxError= max(maxError, abs(y[i]-e));
    printf("Max error:%f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
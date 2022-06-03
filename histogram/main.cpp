#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>
#include <ctime>

#include "DS_timer.h"

DS_timer *timer;
#define TIMER_HOST 0
#define TIMER_KERNEL 1
#define TIMER_KERNEL_SH 2
#define NUM_TIMER (TIMER_KERNEL_SH + 1)

void sequential_Histogram(char* data, int n, int* histo);
__global__ void histo_kernel(char* data, int n, int* histo);
__global__ void histo_shared_kernel(char* data, int n, int* histo, int n_bins);

void setTimer(void);

int main()
{
    srand((unsigned int)time(NULL));
    printf("[Histogram...]\n\n");
    
    timer = NULL;
    setTimer();

    int n = 1 << 24;
    int threads = 256;
    int blocks = 256;
    int n_bins = 7; // the number of bins, a-d, e-h, i-l, m-p, q-t, u-x, y-z

    printf("The number of elements: %d\n", n);
    printf("Threads: %d / Blocks: %d\n\n", threads, blocks);
    
    int dev = 0;
    cudaSetDevice(dev);

    unsigned int bytes = n*sizeof(char);
    char* h_data;
    int* h_histo_host;
    int* h_histo_kernel;
    int* h_histo_kernel_shared;


    // allocate host memory
    h_data = (char*)malloc(bytes);
    h_histo_host = (int*)malloc(n_bins*sizeof(int));
    h_histo_kernel = (int*)malloc(n_bins*sizeof(int));
    h_histo_kernel_shared = (int*)malloc(n_bins*sizeof(int));

    // init
    for (int i = 0; i < n; i++) {
        h_data[i] = 97 + rand() % 26;
    }
    for (int i = 0; i < n_bins; i++){
        h_histo_host[i] = 0;
        h_histo_kernel[i] = 0;
        h_histo_kernel_shared[i] = 0;
    }
        
    // allocate device memory
    char* d_data;
    int *d_histo;
    cudaMalloc((void**)&d_data, bytes);
    cudaMalloc((void**)&d_histo, n_bins*sizeof(int));

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

        timer->onTimer(TIMER_HOST);
            sequential_Histogram(h_data, n, h_histo_host);
        timer->offTimer(TIMER_HOST);
    
        timer->onTimer(TIMER_KERNEL);
            histo_kernel<<<blocks, threads>>>(d_data, n, d_histo);
            cudaMemcpy(h_histo_kernel, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost);
        timer->offTimer(TIMER_KERNEL);

        timer->onTimer(TIMER_KERNEL_SH);
            int smem_size = 2*n_bins*sizeof(int);
            histo_shared_kernel<<<blocks, threads, smem_size>>>(d_data, n, d_histo, 7);
            cudaMemcpy(h_histo_kernel_shared, d_histo, n_bins*sizeof(int), cudaMemcpyDeviceToHost);
        timer->offTimer(TIMER_KERNEL_SH);

    int total_count = 0;
    printf("histo: ");
    for (int i = 0; i < n_bins; i++) {
        printf("%d ", h_histo_kernel[i]);
        total_count += h_histo_kernel[i];
    }
    printf("\n\n");
    printf("Total Count : %d\n", total_count);
    
    timer->printTimer();
    if (timer != NULL)
        delete timer;

    printf(total_count == n ? "Result is correct!\n" : "Result is not correct!\n");

    // free memory
    free(h_data);
    free(h_histo_host);
    free(h_histo_kernel);
    free(h_histo_kernel_shared);
    cudaFree(d_data);
    cudaFree(d_histo);
    return 0;
}

void sequential_Histogram(char* data, int n, int* histo)
{
    for (int i = 0; i < n; i++) {
        int alphabet_pos = data[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26)
            histo[alphabet_pos / 4]++;
    }
}

__global__ void histo_kernel(char* data, int n, int* histo)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int section_size = (n - 1) / (blockDim.x * gridDim.x) + 1;
    int start = i * section_size;

    for (int k = 0; k < section_size; k++) {
        if (start + k < n) {
            int alphabet_pos = data[start + k] - 'a';
            if (alphabet_pos >= 0 && alphabet_pos < 26)
                atomicAdd(&histo[alphabet_pos/4], 1);
        }
    }
}

__global__ void histo_shared_kernel(char* data, int n, int* histo, int n_bins)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Privatized bins
    extern __shared__ int histo_s[];
    if (threadIdx.x < n_bins)
        histo_s[threadIdx.x] = 0u;
    __syncthreads();

    // histogram
    for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
        int alphabet_pos = data[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26)
            atomicAdd(&histo_s[alphabet_pos/4], 1);
    }
    __syncthreads();

    // commit to global memory
    if (threadIdx.x < n_bins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }
}


void setTimer(void) {
    timer = new DS_timer(NUM_TIMER);
    timer->initTimers();
    timer->setTimerName(TIMER_HOST, "CPU code");
    timer->setTimerName(TIMER_KERNEL, "Kernel launch");
    timer->setTimerName(TIMER_KERNEL_SH, "Kernel launch (shared memory)");
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>


const int N = 5;
__device__
const double delta_tau = 0.00, delta_t = 0.01;
#define PI 3.14159265358979323846
__device__
double p = 2000, m = 100, g = 10, v = 0;

__device__
void get_x1(double X2[5], double X0[5], double Ax) {
    X2[0] = (X0[0] + X0[2] * cos(3 * PI / 2 - X0[3]) - Ax);
}

__device__
void get_x2(double X2[5], double X0[5], double Bx) {
    X2[1] = (X0[1] + X0[2] * cos(3 * PI / 2 + X0[4]) - Bx);
}

__device__
void get_y(double* X2, double* X0, double Ay) {
    X2[2] = (X0[2] + X0[2] * sin(3 * PI / 2 - X0[3]) - Ay);
}

__device__
void get_f1(double* X2, double* X0, double C) {
    X2[3] = ((X0[3] + X0[4]) * X0[2] + (X0[1] - X0[0]) - C);
}

__device__
void get_f2(double X2[5], double X0[5], double By) {
    X2[4] = (X0[2] + X0[2] * sin(3 * PI / 2 + X0[4]) - By);
}

__global__
void parallel(double* X0, double* X1, double* X2, double Ax, double Ay, double Bx, double By, double C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    switch (i) {
    case 0:
        get_x1(X2, X0, Ax);
        X1[i] = X0[i] - delta_tau * X2[i];
        break;
    case 1:
        get_x2(X2, X0, Bx);
        X1[i] = X0[i] - delta_tau * X2[i];
        break;
    case 2:
        get_y(X2, X0, Ay);
        X1[i] = X0[i] - delta_tau * X2[i];
        break;
    case 3:
        get_f1(X2, X0, C);
        X1[i] = X0[i] - delta_tau * X2[i];
        break;
    case 4:
        get_f2(X2, X0, By);
        X1[i] = X0[i] - delta_tau * X2[i];
        break;
    }
}

__global__
void serial(double* X0, double* X1, double* X2, double Ax, double Ay, double Bx, double By, double C) {
    get_x1(X2, X0, Ax);
    get_x2(X2, X0, Bx);
    get_y(X2, X0, Ay);
    get_f1(X2, X0, C);
    get_f2(X2, X0, By);
    for (int i = 0; i < 5; i++) {
        X1[i] = X0[i] - delta_tau * X2[i];
    }
}

__global__
void parallelA(double* A0, double* A1, double x1, double x2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%f-%f-%f", A0[0], x1, x2);
    switch (i) {
    case 0:
        A1[0] = A0[0] + A0[1] * delta_t;
        break;
    case 1:
        A1[1] = A0[1] + (p * (x2 - x1) - m * g) / m * delta_t;;
        break;
    }
}

double Ax = -0.353, Ay = 0.3, Bx = 0.353, By = 0.3, C = 3 * PI / 8;

int main() {
    //FILE* ouf = fopen("output.txt", "w");

    double* x0, * x1, * x2, * d_x0, * d_x1, * d_x2, * a0, * a1, * d_a0, * d_a1;
    x0 = (double*)malloc(N * sizeof(double));
    x1 = (double*)malloc(N * sizeof(double));
    x2 = (double*)malloc(N * sizeof(double));
    a0 = (double*)malloc(2 * sizeof(double));
    a1 = (double*)malloc(2 * sizeof(double));

    cudaMalloc(&d_x0, N * sizeof(double));
    cudaMalloc(&d_x1, N * sizeof(double));
    cudaMalloc(&d_x2, N * sizeof(double));
    cudaMalloc(&d_a0, 2 * sizeof(double));
    cudaMalloc(&d_a1, 2 * sizeof(double));

    double x0_init[] = { -0.1, 0.1, 0.0, 2.0, 2.0 };
    memcpy(x0, x0_init, N * sizeof(double));
    a0[0] = 0.3;
    a0[1] = 0;
    double time_sum = 0;
    double time_taken;
    double time_complete_sum = 0;
    cudaMemcpy(d_a0, a0, 2 * sizeof(double), cudaMemcpyHostToDevice);
    for (double t = 0; t <= 0.05; t += delta_t) {
        
        cudaMemcpy(d_x0, x0, N * sizeof(double), cudaMemcpyHostToDevice);
        time_sum = 0;
        for (int step = 0; step < 300000; ++step) {
            double flag = true;
            clock_t start = clock();
            //parallel<<<1, 5>>>(d_x0, d_x1, d_x2, Ax, Ay, Bx, By, C);
            serial << <1, 1 >> > (d_x0, d_x1, d_x2, Ax, Ay, Bx, By, C);
            clock_t end = clock();
            time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            time_sum += time_taken;
            cudaMemcpy(d_x0, d_x1, N * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(x2, d_x2, N * sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 5; i++) { if (x2[i] > 0.0001) { flag = false; } }
            if (flag) { break; }
        }

        cudaMemcpy(x1, d_x1, N * sizeof(double), cudaMemcpyDeviceToHost);
        time_complete_sum += time_sum;
        printf("%f\n", time_sum);
        parallelA << <1, 2 >> > (d_a0, d_a1, x1[0], x1[1]);
        //cudaMemcpy(a1, d_a1, 2 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_a0, d_a1, 2 * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(a1, d_a1, 2 * sizeof(double), cudaMemcpyDeviceToHost);
        Ay = a1[0];
        By = a1[0];
        v = a1[1];

        for (int i = 0; i < 5; ++i) {
            printf("%f ", x1[i]);
        }
        printf("\n");
        //fprintf(ouf, "%f, %f, %f, %f, %f, %f\n", Ax, Ay, Bx, By, C, time_taken);
        //fflush(ouf);
    }
    printf("->%f\n", time_complete_sum/6);
    cudaFree(d_x0);
    cudaFree(d_x1);
    free(x0);
    free(x1);
}

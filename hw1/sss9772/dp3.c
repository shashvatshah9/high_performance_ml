#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mkl_cblas.h>

float bdp(long N, float *pA, float *pB) {
    float R = cblas_sdot(N, pA, 1, pB, 1);
    return R;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Invalid number of args, need 2");
        return 0;
    }
    long n = atol(argv[1]);
    long reps = atol(argv[2]);

    struct timespec *start, *end;

    printf("n:%lu reps:%lu\n", n, reps);

    float *pA, *pB, *res;

    pA = (float *)malloc(sizeof(float) * n);
    pB = (float *)malloc(sizeof(float) * n);
    res = (float *)malloc(sizeof(float) * reps);

    start = (struct timespec *)malloc(sizeof(struct timespec) * reps);
    end = (struct timespec *)malloc(sizeof(struct timespec) * reps);

    // initialize the arrays
    if(pA != NULL && pB != NULL){
        for(int i=0;i<n;i++) {
            pA[i] = 1.0;
            pB[i] = 1.0;
        }
    }

    for (int i = 0; i < reps; i++){
        clock_gettime(CLOCK_MONOTONIC, &start[i]);
        res[i] = bdp(n, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &end[i]);
    }
    printf("%f\n", res[reps-1]);

    // calculate average time for second half of the execution
    double total_half_time = 0.0;
    for(int i=reps/2; i<reps; i++){
        double time_usec=(((double)end[i].tv_sec *1000000 + (double)end[i].tv_nsec/1000)
- ((double)start[i].tv_sec *1000000 + (double)start[i].tv_nsec/1000));
        total_half_time += time_usec;
    }

    double avg_time = total_half_time / (reps/2);
    // 2 floating point elements per op
    double bandwidth = 2*n*sizeof(float)/(8*1000*avg_time); 
    // 2 flops per operation
    double flops_per_sec = 2*n/(1000 * avg_time); 
    double avg_time_secs = avg_time/1000000;

    printf("N: %lu <T>: %.06lf sec B: %0.03lf GB/sec F: %0.3lf GFLOP/sec\n", n, avg_time_secs, bandwidth, flops_per_sec);
    return 0;
}

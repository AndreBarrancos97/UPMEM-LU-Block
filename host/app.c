/**
* app.c
* Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

// Pointer declaration
static T* A_matrix;
static T* U_matrix;
static T* L_matrix;
static T* U_matrix_inv;
static T* A_matrix_inv;
static T* Y_host;

// Create input arrays
static void read_input(T* A, T* B, T* C, T* D, T* E, unsigned int nr_elements) {

    float A_init[64] = {43,7,8,6,4,6,7,3,10,44,3,8,1,10,4,7,1,7,46,7,2,9,8,10,3,1,3,39,8,6,10,3,3,9,10,8,46,7,2,3,10,4,2,10,5,48,9,5,6,1,4,7,2,1,30,4,3,1,7,2,6,6,5,31};
    float U_init[64] = {1.000000, 0.162791, 0.186047, 0.139535, 0.093023, 0.139535, 0.162791, 0.069767,0.000000, 1.000000, 0.026894, 0.155873, 0.001647, 0.203074, 0.055982, 0.148738, 0.000000, 0.000000, 1.000000, 0.126994, 0.041545, 0.163752, 0.163367, 0.195338, 0.000000, 0.000000, 0.000000, 1.000000, 0.199491, 0.133005, 0.237903, 0.058657, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.060212, -0.037906, -0.012936, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.119497, 0.077636, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.101139, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000};
    float U_init_inv[64];
    float A_init_inv[64];

    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 8; j++) {
            U_init_inv[i + 8*j] = U_init[j + 8*i];
            A_init_inv[i + 8*j] = A_init[j + 8*i];
        }   
    }

    for (unsigned int i = 0; i < nr_elements; i++) {
        printf("%f ", U_init_inv[i]);
            if ((i+1)%8 == 0)
                printf("\n");      
        A[i] = A_init[i];
        //B[i] = U_init[i];
        B[i] = 0;
        C[i] = 0;
        //D[i] = U_init_inv[i]; 
        D[i] = 0;
        E[i] = A_init_inv[i];
    }
    printf("\n");   


}

// Compute output in the host for verification purposes
static void axpy_host(T* A, T* B, T alpha, unsigned int nr_elements) {
    for (unsigned int i = 0; i < nr_elements; i++) {
        B[i] = alpha * A[i] + B[i];
        //B[i] =  A[i] + B[i];
    }
}

// Main of the Host Application
int main(int argc, char **argv) {

    // Input parameters
    struct Params p = input_params(argc, argv);

    // Timer declaration
    Timer timer;
#if defined(CYCLES) || defined(INSTRUCTIONS)
    double cc = 0;
    double cc_min = 0;
#endif
	
    // Allocate DPUs
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus)); // Number of DPUs in the DPU set
    printf("Allocated %d DPU(s)\t", nr_of_dpus);
    printf("NR_TASKLETS\t%d\tBLOCK\t%d\n", NR_TASKLETS, BLOCK);

    // Load binary
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    // Input size 
    const unsigned int input_size = p.input_size;         // Total input size 
    printf("input_size =  %d \n",input_size);             //input_size = 10
    printf("sizeof(T) =  %ld \n",sizeof(T));              // int32 = 4bytes

    const unsigned int input_size_8bytes = 
        ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size; // Total input size, 8-byte aligned
    
    printf("input_size_8bytes =  %d \n",input_size_8bytes);

    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)

    printf("input_size_dpu =  %d \n",input_size_dpu);

    const unsigned int input_size_dpu_8bytes = 
        ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned

    printf("input_size_dpu_8bytes =  %d \n",input_size_dpu_8bytes);

    // Input/output allocation in host main memory
    A_matrix = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    U_matrix = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    L_matrix = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    U_matrix_inv = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    A_matrix_inv = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));

    Y_host = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    T *bufferA = A_matrix;
    T *bufferU = U_matrix;
    T *bufferL = L_matrix;
    T *bufferU_inv = U_matrix_inv;
    T *bufferA_inv = A_matrix_inv;

    T alpha = p.alpha;
    unsigned int i = 0;

    // Create an input file with arbitrary data
    read_input(A_matrix, U_matrix, L_matrix, U_matrix_inv, A_matrix_inv, input_size);
    memcpy(Y_host, U_matrix, input_size_dpu_8bytes * nr_of_dpus * sizeof(T));

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Compute output on CPU (verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        //axpy_host(X, Y_host, alpha, input_size);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        printf("Load input data\n");
        // Input arguments
        unsigned int kernel = 0;
        dpu_arguments_t input_arguments[NR_DPUS];
        for(i=0; i<nr_of_dpus-1; i++) {
            input_arguments[i].size=input_size_dpu_8bytes * sizeof(T); 
            input_arguments[i].transfer_size=input_size_dpu_8bytes * sizeof(T); 
            input_arguments[i].kernel=kernel;
            input_arguments[i].alpha=alpha;
        }
        input_arguments[nr_of_dpus-1].size=(input_size_8bytes - input_size_dpu_8bytes * (NR_DPUS-1)) * sizeof(T); 
        input_arguments[nr_of_dpus-1].transfer_size=input_size_dpu_8bytes * sizeof(T); 
        input_arguments[nr_of_dpus-1].kernel=kernel;
        input_arguments[nr_of_dpus-1].alpha=alpha;

        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup); // Start timer (CPU-DPU transfers)
        i = 0;
		// Copy input arguments
        // Parallel transfers
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

        // Copy input arrays
#ifdef SERIAL // Serial transfers

        //@@ INSERT SERIAL CPU-DPU TRANSFER HERE

#else // Parallel transfers

        //@@ INSERT PARALLEL CPU-DPU TRANSFER HERE
        DPU_FOREACH(dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferA + input_size_dpu_8bytes * i));
        }

        DPU_ASSERT (dpu_push_xfer (dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
 
        DPU_FOREACH (dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferU + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));

        DPU_FOREACH (dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferL + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T)*2, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));

        DPU_FOREACH (dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferU_inv + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T)*3, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT)); 

        DPU_FOREACH (dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferA_inv + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T)*4, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));          

#endif
        if(rep >= p.n_warmup)
            stop(&timer, 1); // Stop timer (CPU-DPU transfers)
		
        printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup); // Start timer (DPU kernel)
        }
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= p.n_warmup) {
            stop(&timer, 2); // Stop timer (DPU kernel)
        }
        
        //DPU_FOREACH (dpu_set, dpu) {
        //DPU_ASSERT(dpu_log_read(dpu, stdout));
        //}


#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif

        printf("Retrieve results\n");
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup); // Start timer (DPU-CPU transfers)
        i = 0;
        // Copy output array
#ifdef SERIAL // Serial transfers

        //@@ INSERT SERIAL DPU-CPU TRANSFER HERE

#else // Parallel transfers

        //@@ INSERT PARALLEL DPU-CPU TRANSFER HERE
        /*dpu_results_t results[nr_of_dpus];
        // Parallel transfers
        dpu_results_t* results_retrieve[nr_of_dpus];
        DPU_FOREACH(dpu_set, dpu, i) {
            results_retrieve[i] = (dpu_results_t*)malloc(NR_TASKLETS * sizeof(dpu_results_t));
            DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, NR_TASKLETS * sizeof(dpu_results_t), DPU_XFER_DEFAULT));
        DPU_FOREACH(dpu_set, dpu, i) {
            results[i].count = 0;
            // Retrieve tasklet count
            for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
                if (results_retrieve[i][each_tasklet].count > results[i].count)
                    results[i].count = results_retrieve[i][each_tasklet].count;
            }
            //free(results_retrieve[i]);
        }*/
        
        DPU_FOREACH(dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferL + input_size_dpu_8bytes * i));
        }

        DPU_ASSERT (dpu_push_xfer (dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,  input_size_dpu_8bytes * sizeof(T) * 2, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        
        DPU_FOREACH(dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferU_inv + input_size_dpu_8bytes * i));
        }

        DPU_ASSERT (dpu_push_xfer (dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,  input_size_dpu_8bytes * sizeof(T) * 3, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        

#endif
        if(rep >= p.n_warmup)
            stop(&timer, 3); // Stop timer (DPU-CPU transfers)

#if defined(CYCLES) || defined(INSTRUCTIONS)
        dpu_results_t results[nr_of_dpus];
        // Parallel transfers
        dpu_results_t* results_retrieve[nr_of_dpus];
        DPU_FOREACH(dpu_set, dpu, i) {
            results_retrieve[i] = (dpu_results_t*)malloc(NR_TASKLETS * sizeof(dpu_results_t));
            DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, NR_TASKLETS * sizeof(dpu_results_t), DPU_XFER_DEFAULT));
        DPU_FOREACH(dpu_set, dpu, i) {
            results[i].count = 0;
            // Retrieve tasklet count
            for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
                if (results_retrieve[i][each_tasklet].count > results[i].count)
                    results[i].count = results_retrieve[i][each_tasklet].count;
            }
            free(results_retrieve[i]);
        }

        uint64_t max_count = 0;
        uint64_t min_count = 0xFFFFFFFFFFFFFFFF;
        // Print performance results
        if(rep >= p.n_warmup){
            i = 0;
            DPU_FOREACH(dpu_set, dpu) {
                if(results[i].count > max_count)
                    max_count = results[i].count;
                if(results[i].count < min_count)
                    min_count = results[i].count;
                i++;
            }
            cc += (double)max_count;
            cc_min += (double)min_count;
        }
#endif
    }
#ifdef CYCLES
    printf("DPU cycles  = %g\n", cc / p.n_reps);
#elif INSTRUCTIONS
    printf("DPU instructions  = %g\n", cc / p.n_reps);
#endif
	
    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 3, p.n_reps);
    printf("\n");
    // Check output
    bool status = true;
    printf("********* Lower Matrix ******************** \n");
    for (i = 0; i < input_size; i++) {
        //printf("%d: %f -- %f\n", i, Y_host[i], L_matrix[i]);
        printf("%f ",L_matrix[i]);
        if((i+1)%8 == 0){
            printf("\n");
        }
        if(Y_host[i] != U_matrix[i]){ 
            status = false;
            //printf("%d: ******** %f ************\n", i, L_matrix[i]);
        }
    }
    printf(" \n ********* Lower Matrix ******************** \n");
   
        float U_init_inv[64];
        for (unsigned int i = 0; i < 8; i++) {
            for (unsigned int j = 0; j < 8; j++) {
                U_init_inv[i + 8*j] = bufferU_inv[j + 8*i];
            }   
        }
    printf("\n ********* Upper Matrix ******************** \n");
    for (i = 0; i < input_size; i++) {
        //printf("%d: %f -- %f\n", i, Y_host[i], L_matrix[i]);
        printf("%f ",U_init_inv[i]);
        if((i+1)%8 == 0){
            printf("\n");
        }
    }
    printf("\n ********* Upper Matrix ******************** \n");   
    printf("\n");
    printf("\n ********* AUX Matrix ******************** \n");
	for (int i = 0; i < 8; ++i){
		for (int i_aux = 0; i_aux < 8; ++i_aux){
			float aux_v2 = 0;
			for (int j = 0; j < 8; ++j){
				aux_v2 = aux_v2 + L_matrix[i*8 + j]*U_init_inv[j*8 + i_aux];
                
			}
            printf("%f ", aux_v2);
			//a_aux[i][i_aux] = aux_v2;
		}
        printf("\n");
	}    
    printf("\n");
    printf("\n ********* AUX Matrix ******************** \n");



    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A_matrix);
    free(U_matrix);
    free(L_matrix);
    free(Y_host);
    DPU_ASSERT(dpu_free(dpu_set)); // Deallocate DPUs
	
    return status ? 0 : -1;
}

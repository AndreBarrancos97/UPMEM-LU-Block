#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <math.h>

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

// Print Matrix
void print_matrix_2D(float matrix[], int size, int line)
{
	int j = 0;
	for (int i = 0; i < size; i++){

		if ((i%line) == 0){
			printf("\n");
			j=0;
		}

		printf("%f ",matrix[i]);
		j++;
	}
}

// Read line size of loaded matrix
static void read_size(const char *filename, unsigned int *size){
    FILE *file = fopen(filename, "r");
        if (file == NULL) {
            perror("Unable to open file for reading");
        }
    
    fscanf(file, "%d", size);

    fclose(file);

}

// Create input arrays
static void read_input( T* A, const char *filename, int nr_elements){
    //float A_init[64] = {48, 7, 8, 6, 4, 6, 7, 3, 10, 49, 3, 8, 1, 10, 4, 7, 1, 7, 51, 7, 2, 9, 8, 10, 3, 1, 3, 44, 8, 6, 10, 3, 3, 9, 10, 8, 46, 7, 2, 3,  10, 4, 2, 10, 5, 53, 9, 5, 6, 1, 4, 7, 2, 1, 30, 4, 3, 1, 7, 2, 6, 6, 5, 36};
    FILE *file = fopen(filename, "r");
        if (file == NULL) {
            perror("Unable to open file for reading");
        }
   
    fscanf(file, "%f", &A[0]);

    for (int i = 0; i < (nr_elements); i++) {

        fscanf(file, "%f", &A[i]);
        
    }
    fclose(file);

}

// Main of the Host Application
int main(int argc, char **argv) {
    // Input parameters
    struct Params p = input_params(argc, argv);
    
    // Alterar o filename e o Makefile
    // Nr tasklets = Nr de linhas da matriz
    // 2^BLOCK = Nr de elementos na linha * 4 bytes. Ex: BLOCK = 5 -> 2^5 = 64 = 16 elmentos numa linha * 4 bytes.
    //char filename[] = "matrix_4x4_16.txt";
    char filename[] = "matrix_8x8_64.txt";
    unsigned int input_line_size, input_size;

    read_size(filename, &input_line_size);

    input_size = input_line_size*input_line_size;
    printf("input_size =  %d \n",input_size);

    // Timer declaration
    Timer timer;
	
    // Allocate DPUs
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));

    // Number of DPUs in the DPU set
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));                                                                      
    printf("Allocated %d DPU(s)\t", nr_of_dpus);
    printf("NR_TASKLETS\t%d\tBLOCK\t%d\n", NR_TASKLETS, BLOCK);

    // Load binary
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    // Input size 
    // Total input size = 64 (8x8 matrix)
    // Input_size = 10
    // Float = 4bytes
    // Const unsigned int input_size = p.input_size;
    printf("input_size =  %d \n",input_size);                                                                               
    printf("sizeof(T) =  %ld \n",sizeof(T));                                                                                

    // Total input size, 8-byte aligned
    const unsigned int input_size_8bytes = 
        ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size;                                          
    printf("input_size_8bytes =  %d \n",input_size_8bytes);

    // Input size per DPU (max.)
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus);                                                    
    printf("input_size_dpu =  %d \n",input_size_dpu);

    // Input size per DPU (max.), 8-byte aligned
    const unsigned int input_size_dpu_8bytes = 
        ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu;                              
    printf("input_size_dpu_8bytes =  %d \n",input_size_dpu_8bytes);

    // Input/output allocation in host main memory
    A_matrix = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    U_matrix = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    L_matrix = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));

    T *bufferA = A_matrix;
    T *bufferU = U_matrix;
    T *bufferL = L_matrix;

    //T alpha = p.alpha;
    uint32_t i_index = 0;
    uint32_t code_part = 0;
    unsigned int i = 0;

    // Create an input file with arbitrary data
    read_input(A_matrix, filename, input_size);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        printf("rep = %d \n", rep);
        printf("p.n_warmup = %d \n", p.n_warmup);
        printf("p.n_reps = %d \n", p.n_reps);
        // Compute output on CPU (verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);

        if(rep >= p.n_warmup)
            stop(&timer, 0);

        printf("Load input data\n");
        for(unsigned int j = 0; j < (input_line_size*2);j++){
        // Input arguments
        unsigned int kernel = 0;
        dpu_arguments_t input_arguments[NR_DPUS];
        for(i=0; i<nr_of_dpus-1; i++) {
            //input_arguments[i].size=input_size_dpu_8bytes * sizeof(T); 
            //input_arguments[i].transfer_size=input_size_dpu_8bytes * sizeof(T);
            input_arguments[i].size=input_size_8bytes * sizeof(T); 
            input_arguments[i].transfer_size=input_size_8bytes * sizeof(T);  
            input_arguments[i].kernel=kernel;
            input_arguments[i].i_index=i_index;
            input_arguments[i].code_part=code_part;
            input_arguments[i].dpu_nr=i;
            input_arguments[i].tasklet_nr=NR_TASKLETS;
        }

        //input_arguments[nr_of_dpus-1].size=(input_size_8bytes - input_size_dpu_8bytes * (NR_DPUS-1)) * sizeof(T); 
        //input_arguments[nr_of_dpus-1].transfer_size=input_size_dpu_8bytes * sizeof(T);
        input_arguments[i].size=input_size_8bytes * sizeof(T); 
        input_arguments[i].transfer_size=input_size_8bytes * sizeof(T);  
        input_arguments[nr_of_dpus-1].kernel=kernel;
        input_arguments[i].i_index=i_index;
        input_arguments[i].code_part=code_part;
        input_arguments[i].dpu_nr =i;
        input_arguments[i].tasklet_nr=NR_TASKLETS;

        // Start timer (CPU-DPU transfers)
        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup); 
        i = 0;

		// Copy input arguments
        // Parallel transfers
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

        // Parallel transfers
        // Copy matrixes
        DPU_FOREACH(dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferA + input_size_dpu_8bytes * 0));
        }
        DPU_ASSERT (dpu_push_xfer (dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT));
 
        DPU_FOREACH (dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferL + input_size_dpu_8bytes * 0));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_8bytes * sizeof(T)*1, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT));

        DPU_FOREACH (dpu_set, dpu, i) {
	        DPU_ASSERT (dpu_prepare_xfer (dpu, bufferU + input_size_dpu_8bytes * 0));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_8bytes * sizeof(T)*2, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT)); 

        // Stop timer (CPU-DPU transfers)
        if(rep >= p.n_warmup)
            stop(&timer, 1); 
		
        // Run DPU kernel
        printf("Run program on DPU(s) \n");
        
        // Start timer (DPU kernel)
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup); 
        }

        // Stop timer (DPU kernel)
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= p.n_warmup) {
            stop(&timer, 2); 
        }

        unsigned int each_dpu = 0;
        printf("Display DPU Logs\n");
        DPU_FOREACH (dpu_set, dpu) {
            printf("DPU#%d:\n", each_dpu);
            DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
            each_dpu++;
        }

        printf("Retrieve results\n");

        // Start timer (DPU-CPU transfers)
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup); 
        i = 0;
        
        // Copy output array
        // Parallel transfers
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT (dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T) * (nr_of_dpus + i), bufferL + input_size_dpu_8bytes * i,input_size_dpu_8bytes * sizeof(T)));
            DPU_ASSERT (dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T) * ((nr_of_dpus*2) + i), bufferU + input_size_dpu_8bytes * i,input_size_dpu_8bytes * sizeof(T)));
        }
        // Stop timer (DPU-CPU transfers)
        if(rep >= p.n_warmup)
            stop(&timer, 3);

        if ((j+1)%2 == 0){
            i_index = i_index + 1;
            
        }
        if (code_part == 0){
            code_part = 1;
        }
        else{
            code_part = 0;
        } 
        }
    }
	
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
    printf(" \n ********* Lower Matrix ******************** \n");
    for (i = 0; i < input_size; i++) {
        printf("%f ",L_matrix[i]);
        if((i+1)%input_line_size == 0){
            printf("\n");
        }
    }
    printf("********* Lower Matrix ******************** \n");
   
        float U_init_inv[input_size];
        for (unsigned int i = 0; i < input_line_size; i++) {
            for (unsigned int j = 0; j < input_line_size; j++) {
                U_init_inv[i + input_line_size*j] = bufferU[j + input_line_size*i];
            }   
        }

    printf("\n ********* Upper Matrix ******************** \n");
    for (i = 0; i < input_size; i++) {
        printf("%f ",U_init_inv[i]);
        if((i+1)%input_line_size == 0){
            printf("\n");
        }
    }
    printf("********* Upper Matrix ******************** \n");   

    printf(" \n ********* Original A Matrix ******************** \n");
    for (i = 0; i < input_size; i++) {
        printf("%f ",A_matrix[i]);
        if((i+1)%input_line_size == 0){
            printf("\n");
        }
    }
    printf("********* Original A Matrix ******************** \n");

    printf("\n ********* L*U Matrix ******************** \n");
	for (unsigned int i = 0; i < input_line_size; ++i){
		for (unsigned int i_aux = 0; i_aux < input_line_size; ++i_aux){
			float aux_v2 = 0;
			for (unsigned int j = 0; j < input_line_size; ++j){
                
				aux_v2 = aux_v2 + L_matrix[i*input_line_size + j]*U_init_inv[j*input_line_size + i_aux];
			}
            printf("%f ", aux_v2);
            if (fabs(A_matrix[i*input_line_size + i_aux] - aux_v2) > 0.001){
                status = false;
                //printf(" \n *** yooooo = %f *** \n ",aux_v2);
            }
        }		
        printf("\n");
	}    
    printf("********* L*U Matrix ******************** \n");


    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A_matrix);
    free(U_matrix);
    free(L_matrix);
    DPU_ASSERT(dpu_free(dpu_set)); // Deallocate DPUs
	
    return status ? 0 : -1;
}

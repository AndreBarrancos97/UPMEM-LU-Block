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
static T* U_init_inv;

// Print Matrix
void print_matrix_2D(T *matrix1, int input_size, int input_line_size, int lu_multiply, bool *status){
    
    if (lu_multiply == 0){
        for (int i = 0; i < input_size; i++) {
            printf("%f ",matrix1[i]);
            if((i+1)%input_line_size == 0){
                printf("\n");
            }
        }
    }
    else {
        for (int i = 0; i < input_line_size; ++i){
            for (int i_aux = 0; i_aux < input_line_size; ++i_aux){
                float aux_v2 = 0;
                for (int j = 0; j < input_line_size; ++j){
                    
                    aux_v2 = aux_v2 + L_matrix[i*input_line_size + j]*U_init_inv[j*input_line_size + i_aux];
                }
                printf("%f ", aux_v2);
                if (fabs(A_matrix[i*input_line_size + i_aux] - aux_v2) > 0.1){
                    *status = false;
                }
            }		
            printf("\n");
        }
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

// Read matrix from file
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
    // Change filename and Makefile
    // Nr tasklets = Nr of matrix lines
    // 2^BLOCK / 4 bytes = Nr of matrix lines. Ex: BLOCK = 5 -> 2^5 = 32 bytes / 4 bytes = 8 nr of lines.
    
    //char filename[] = "matrix_4x4.txt";
    //char filename[] = "matrix_8x8.txt";
    //char filename[] = "matrix_16x16.txt";
    //char filename[] = "matrix_32x32.txt";
    //char filename[] = "matrix_64x64.txt";
    //char filename[] = "matrix_128x128.txt";
    char filename[] = "matrix_256x256.txt";
    //char filename[] = "matrix_512x512.txt";
    
    unsigned int input_line_size, input_size;
    read_size(filename, &input_line_size);

    input_size = input_line_size * input_line_size;
    printf("input_size =  %d \n",input_size);

    // Input parameters
    struct Params p = input_params(argc, argv);

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
    U_init_inv = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));;

    T *bufferA = A_matrix;
    T *bufferU = U_matrix;
    T *bufferL = L_matrix;
    T *bufferUinv = U_init_inv;

    //T alpha = p.alpha;
    uint32_t i_index = 0;
    uint32_t code_part = 0;
    unsigned int i = 0;

    // Create an input file with arbitrary data
    read_input(A_matrix, filename, input_size);

    start(&timer, 0, 0);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Loop runs two times more. First time calculate L matrix. Second time calculate U matrix.
        for(unsigned int j = 0; j < (input_line_size*2);j++){
            
            // Input arguments
            unsigned int kernel = 0;
            dpu_arguments_t input_arguments[NR_DPUS];
            
            for(i=0; i<nr_of_dpus-1; i++) {
                input_arguments[i].size = input_size_8bytes * sizeof(T); 
                input_arguments[i].transfer_size = input_size_8bytes * sizeof(T);  
                input_arguments[i].kernel = kernel;
                input_arguments[i].i_index = i_index;
                input_arguments[i].code_part = code_part;
                input_arguments[i].dpu_nr = i;
                input_arguments[i].tasklet_nr = NR_TASKLETS;
            }
            input_arguments[i].size = input_size_8bytes * sizeof(T); 
            input_arguments[i].transfer_size = input_size_8bytes * sizeof(T);  
            input_arguments[nr_of_dpus-1].kernel = kernel;
            input_arguments[i].i_index = i_index;
            input_arguments[i].code_part = code_part;
            input_arguments[i].dpu_nr = i;
            input_arguments[i].tasklet_nr = NR_TASKLETS;
 
            // Copy input arguments
            // Parallel transfers
            DPU_FOREACH(dpu_set, dpu, i) {
                DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

            // Parallel transfers
            // Copy matrixes
            DPU_FOREACH(dpu_set, dpu, i) {
                DPU_ASSERT (dpu_prepare_xfer (dpu, bufferA));
            }
            DPU_ASSERT (dpu_push_xfer (dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT));
    
            DPU_FOREACH (dpu_set, dpu, i) {
                DPU_ASSERT (dpu_prepare_xfer (dpu, bufferL));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_8bytes * sizeof(T)*1, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT));

            DPU_FOREACH (dpu_set, dpu, i) {
                DPU_ASSERT (dpu_prepare_xfer (dpu, bufferU));
            }
            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_8bytes * sizeof(T)*2, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT)); 
            
            /*// Broadcast transfer is slower
            DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, 0, bufferA, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT));
            DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, input_size_8bytes * sizeof(T)*1, bufferL, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT));
            DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, input_size_8bytes * sizeof(T)*2, bufferU, input_size_8bytes * sizeof(T), DPU_XFER_DEFAULT));
            */
   
            // Run DPU kernel
            // Run program on DPU(s)
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

            // Display DPU logs
            /*
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
            */
                    
            // Retrieve results. Serial transfers.
            DPU_FOREACH(dpu_set, dpu, i) {
                DPU_ASSERT (dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T) * (nr_of_dpus + i), bufferL + input_size_dpu_8bytes * i,input_size_dpu_8bytes * sizeof(T)));
                DPU_ASSERT (dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T) * ((nr_of_dpus*2) + i), bufferU + input_size_dpu_8bytes * i,input_size_dpu_8bytes * sizeof(T)));
            }

            // Every two loops the first L and U lines are calculated. So change line, which is indexed by i_index. 
            if ((j+1)%2 == 0){
                i_index = i_index + 1;
                
            }

            // Finishing running the first part of the code. L line matrix.
            if (code_part == 0){
                code_part = 1;
            }

            // Finishing running the second part of the code. U line matrix.
            else{
                code_part = 0;
            } 
        }
    }

	stop(&timer, 0);
    
    printf("\n");
    
    // Check output
    bool status = true;
    printf(" \n ********* Lower Matrix ******************** \n");
    
    print_matrix_2D(bufferL, input_size, input_line_size,0,&status);                                    // Print L matrix

    printf("********* Lower Matrix ******************** \n");
    
    // Invert U matrix
    for (unsigned int i = 0; i < input_line_size; i++) {
        for (unsigned int j = 0; j < input_line_size; j++) {
            U_init_inv[i + input_line_size*j] = bufferU[j + input_line_size*i];
        }   
    }

    printf("\n ********* Upper Matrix ******************** \n");
    
    print_matrix_2D(U_init_inv, input_size, input_line_size,0,&status);                                 // Print U matrix
    
    printf("********* Upper Matrix ******************** \n");   

    printf(" \n ********* Original A Matrix ******************** \n");
    
    print_matrix_2D(bufferA, input_size, input_line_size,0,&status);                                    // Print A matrix
    
    printf("********* Original A Matrix ******************** \n");

    printf("\n ********* L*U Matrix ******************** \n");
    
    print_matrix_2D(bufferA, input_size, input_line_size,1,&status);                                    // bufferA is ignored
    
    printf("********* L*U Matrix ******************** \n");

    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Print timing results
    printf("Program finished: ");
    print(&timer, 0, p.n_reps);
    printf("\n");
    // Deallocation
    free(A_matrix);
    free(U_matrix);
    free(L_matrix);
    DPU_ASSERT(dpu_free(dpu_set)); // Deallocate DPUs
	
    return status ? 0 : -1;
}

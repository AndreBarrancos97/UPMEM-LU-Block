#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"
#include "../support/cyclecount.h"

// Input and output arguments
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { 
    // Kernel
    return kernels[DPU_INPUT_ARGUMENTS.kernel](); 
}

static void calc_L_matrix(T *bufferL, T *bufferU_aux, T *bufferA, unsigned int j, unsigned int i) {
    

		if (j < i)
		{
		    bufferL[i] = 0;
        }
        else
        {
            bufferL[i] = bufferA[i];
            for (unsigned int k = 0; k < i; k++)
            {
                // Deduct from the current l cell the value of these 2 values multiplied
                bufferL[i] = bufferL[i] - bufferL[k] * bufferU_aux[k];
            }
        }
}

static void calc_U_matrix(T *bufferL_aux, T *bufferU, T *bufferA_inv, unsigned int j, unsigned int i) {
    
		if (j < i)
		{
		    bufferU[i] = 0;
        }
		else if (j == i)
		{
		    bufferU[i] = 1;
        }
        else
        {
            bufferU[i] = bufferA_inv[j]/bufferL_aux[i];
            for (unsigned int k = 0; k < i; k++)
            {
                // Deduct from the current l cell the value of these 2 values multiplied
                bufferU[i] = bufferU[i] - ((bufferL_aux[k] * bufferU[k])/bufferL_aux[i]);
            }
        }
}

// main_kernel1
int main_kernel1() {
    
    unsigned int tasklet_id = me();

    if (tasklet_id == 0){
        // Reset the heap
        mem_reset(); 
    }

    // Barrier
    barrier_wait(&my_barrier);

    // Input size per DPU in bytes
    // Transfer input size per DPU in bytes
    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size;                                                                               
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size;                                                             
    uint32_t i_index = DPU_INPUT_ARGUMENTS.i_index;                                                           
    uint32_t code_part = DPU_INPUT_ARGUMENTS.code_part;  
    uint32_t dpu_nr = DPU_INPUT_ARGUMENTS.dpu_nr;  
    uint32_t tasklet_nr = DPU_INPUT_ARGUMENTS.tasklet_nr;

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = (tasklet_id + (dpu_nr*tasklet_nr)) << BLOCK_SIZE_LOG2;
    uint32_t base_tasklet1 = (tasklet_id + (dpu_nr*tasklet_nr));
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_L = (uint32_t)(DPU_MRAM_HEAP_POINTER + (input_size_dpu_bytes_transfer));
    uint32_t mram_base_addr_U = (uint32_t)(DPU_MRAM_HEAP_POINTER + (input_size_dpu_bytes_transfer*2));

    /*
    printf("input_size_dpu_bytes [%d] =  %d \n",tasklet_id, input_size_dpu_bytes);
    printf("input_size_dpu_bytes_transfer [%d] =  %d \n",tasklet_id, input_size_dpu_bytes_transfer);
    printf("base_tasklet [%d] =  %d \n",tasklet_id, base_tasklet);
    printf("BLOCK_SIZE_LOG2 [%d] =  %d \n",tasklet_id, BLOCK_SIZE_LOG2);
    printf("BLOCK_SIZE [%d] =  %d \n",tasklet_id, BLOCK_SIZE);
    */

    //printf("tasklet_i[%d] = %d &&&& %d \n", tasklet_id, base_tasklet, base_tasklet1);
    printf("tasklet_i[%d] = %d \n", tasklet_id, BLOCK_SIZE);

    // Initialize a local cache in WRAM to store the MRAM block
    T *cache_A = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_U = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_L = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_aux = (T *) mem_alloc(BLOCK_SIZE);

    /*
    Example:
        input_size = 64;
        BLOCK = 5;
        BLOCK_SIZE_LOG2 = BLOCK;
        BLOCK_SIZE = (1 << BLOCK_SIZE_LOG2) = 32 bytes;

        input_size_dpu_bytes = input_size * sizeof(T) = 256 bytes;

        To analyse all of these data I need 8 tasklets = input_size_dpu_bytes/BLOCK_SIZE

        However, for example, if I only use 7 tasklets. 
        The tasklet with ID=0 will get the first and last chunk of data.
                                        Positions to read from the MRAM to the WRAM per tasklet
        base_tasklet [ID tasklet = 0] =                         0
        base_tasklet [ID tasklet = 1] =                         32 
        base_tasklet [ID tasklet = 2] =                         64
        base_tasklet [ID tasklet = 3] =                         96
        base_tasklet [ID tasklet = 4] =                         128
        base_tasklet [ID tasklet = 5] =                         160
        base_tasklet [ID tasklet = 6] =                         192
        base_tasklet [ID tasklet = 0] =                         224 
    */

    
	// Each tasklet outputs a line for the Lower Matrix and a column for the upper matrix.
    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){

        // Bound checking
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

        unsigned int j = tasklet_id + (dpu_nr * tasklet_nr);
        unsigned int in_byte_index =  (l_size_bytes * j);
        
        /*
        printf("byte_index [%d] =  %d \n",tasklet_id, in_byte_index);
        printf("l_size_bytes [%d] =  %d \n",tasklet_id, l_size_bytes);
        printf("byte_index [%d] =  %d \n",tasklet_id, byte_index);
        */

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const*) (mram_base_addr_A + in_byte_index), cache_A, l_size_bytes);                              // Load one line from Matrix A
        mram_read((__mram_ptr void const*) (mram_base_addr_U + (l_size_bytes*i_index)), cache_aux, l_size_bytes);                   // Load one indexed line from Matrix U
        
        calc_L_matrix(cache_L, cache_aux, cache_A, j, i_index);
        
        mram_write(cache_L, (__mram_ptr void*) (mram_base_addr_L + in_byte_index), l_size_bytes);                                   // Save one line from Matrix L
            
        // First section of the code. Calculate the L part for each line.
        if (code_part == 0){
            return 0;
        }     

        mram_read((__mram_ptr void const*) (mram_base_addr_A + (l_size_bytes*i_index)), cache_A, l_size_bytes);                     // Load one col from Matrix A
        mram_read((__mram_ptr void const*) (mram_base_addr_L + (l_size_bytes*i_index)), cache_aux, l_size_bytes);                   // Load one line from Matrix L
        
        calc_U_matrix(cache_aux, cache_U, cache_A, j, i_index);
        
        mram_write(cache_U, (__mram_ptr void*) (mram_base_addr_U + in_byte_index), l_size_bytes);                                   // Save one line from Matrix U

        // Second section of the code. Calculate the U part for each line.
        if (code_part == 1){
            return 0;
        }                                                                                     
        
    }
	
    return 0;
}

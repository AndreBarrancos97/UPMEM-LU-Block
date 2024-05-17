/*
* AXPY with multiple tasklets
*
*/
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
                //deduct from the current l cell the value of these 2 values multiplied
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
            bufferU[i] = bufferA_inv[i]/bufferL_aux[i];
            for (unsigned int k = 0; k < i; k++)
            {
                //deduct from the current l cell the value of these 2 values multiplied
                bufferU[i] = bufferU[i] - ((bufferL_aux[k] * bufferU[k])/bufferL_aux[i]);
            }
        }
}

// main_kernel1
int main_kernel1() {
    
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0){ 
        mem_reset(); // Reset the heap
#ifdef CYCLES
        perfcounter_config(COUNT_CYCLES, true); // Initialize once the cycle counter
#elif INSTRUCTIONS
        perfcounter_config(COUNT_INSTRUCTIONS, true); // Initialize once the instruction counter
#endif
    }
    // Barrier
    barrier_wait(&my_barrier);
#if defined(CYCLES) || defined(INSTRUCTIONS)
    perfcounter_count count;
    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
    result->count = 0;
    counter_start(&count); // START TIMER
#endif

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size;                                                                               // Input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size;                                                             // Transfer input size per DPU in bytes

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_L = (uint32_t)(DPU_MRAM_HEAP_POINTER + (input_size_dpu_bytes_transfer));
    uint32_t mram_base_addr_U = (uint32_t)(DPU_MRAM_HEAP_POINTER + (input_size_dpu_bytes_transfer*2));
    uint32_t mram_base_addr_A_inv = (uint32_t)(DPU_MRAM_HEAP_POINTER + (input_size_dpu_bytes_transfer*3));

    /*
    printf("input_size_dpu_bytes [%d] =  %d \n",tasklet_id, input_size_dpu_bytes);
    printf("input_size_dpu_bytes_transfer [%d] =  %d \n",tasklet_id, input_size_dpu_bytes_transfer);
    printf("base_tasklet [%d] =  %d \n",tasklet_id, base_tasklet);
    printf("BLOCK_SIZE_LOG2 [%d] =  %d \n",tasklet_id, BLOCK_SIZE_LOG2);
    printf("BLOCK_SIZE [%d] =  %d \n",tasklet_id, BLOCK_SIZE);
    */

    // Initialize a local cache in WRAM to store the MRAM block
    T *cache_A = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_U = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_L = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_L_aux = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_U_aux = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_A_inv = (T *) mem_alloc(BLOCK_SIZE);   

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

    
	// each tasklet outputs a line for the Lower Matrix and a column for the upper matrix.
    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){

        // Bound checking
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const*) (mram_base_addr_A + byte_index), cache_A, l_size_bytes);                                         // Read CacheA line                                        
        mram_read((__mram_ptr void const*) (mram_base_addr_A_inv + byte_index), cache_A_inv, l_size_bytes);                                 // Read CacheA column

        unsigned int j = tasklet_id;
        for(unsigned int i = 0; i<8;i++){
            mram_read((__mram_ptr void const*) (mram_base_addr_U + (l_size_bytes*i)), cache_U_aux, l_size_bytes);                            // Read CacheU aux column
            calc_L_matrix(cache_L, cache_U_aux, cache_A, j, i);
            mram_write(cache_L, (__mram_ptr void*) (mram_base_addr_L + byte_index), l_size_bytes);                                           // Save CacheL line
            mram_read((__mram_ptr void const*) (mram_base_addr_L + byte_index), cache_L, l_size_bytes);                                      // Read CacheL line
            
            barrier_wait(&my_barrier);                                                                                                       // Wait for all the tasklets

            mram_read((__mram_ptr void const*) (mram_base_addr_L + (l_size_bytes*i)), cache_L_aux, l_size_bytes);                            // Read Cache L aux line
            calc_U_matrix(cache_L_aux, cache_U, cache_A_inv, j, i);
            mram_write(cache_U, (__mram_ptr void*) (mram_base_addr_U + byte_index), l_size_bytes);                                           // Save CacheU line
            mram_read((__mram_ptr void const*) (mram_base_addr_U + byte_index), cache_U, l_size_bytes);                                      // Read CacheU column

            barrier_wait(&my_barrier);                                                                                                       // Wait for all the tasklets
        } 
    }

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}

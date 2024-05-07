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

// AXPY: Computes AXPY for a cached block 
static void axpy(T *bufferY, T *bufferX, T alpha, unsigned int l_size) {

    //@@ INSERT AXPY CODE
    printf("l_size = %d \n",l_size);
    for (unsigned int i = 0; i < l_size; i++) {
        bufferY[i] = ( bufferX[i]);
        //printf("bufferY[i] = %d \n",bufferY[i]);
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

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size; // Transfer input size per DPU in bytes
    T alpha = DPU_INPUT_ARGUMENTS.alpha; // alpha (a in axpy)
    printf("bbbbb %d \n", DPU_INPUT_ARGUMENTS.alpha);

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_X = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_Y = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);

    // Initialize a local cache in WRAM to store the MRAM block
    
    printf("input_size_dpu_bytes [%d] =  %d \n",tasklet_id, input_size_dpu_bytes);
    printf("input_size_dpu_bytes_transfer [%d] =  %d \n",tasklet_id, input_size_dpu_bytes_transfer);
    printf("base_tasklet [%d] =  %d \n",tasklet_id, base_tasklet);
    printf("BLOCK_SIZE_LOG2 [%d] =  %d \n",tasklet_id, BLOCK_SIZE_LOG2);
    printf("BLOCK_SIZE [%d] =  %d \n",tasklet_id, BLOCK_SIZE);
    

    T *cache_X = (T *) mem_alloc(BLOCK_SIZE);   
    T *cache_Y = (T *) mem_alloc(BLOCK_SIZE);  

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

    
	
    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){

        // Bound checking
        //@@ INSERT BOUND CHECKING HERE
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;
        //printf("l_size_bytes = %d \n ", l_size_bytes);
        // Load cache with current MRAM block
        //@@ INSERT MRAM-WRAM TRANSFERS HERE
        mram_read((__mram_ptr void const*) (mram_base_addr_X + byte_index), cache_X, l_size_bytes); 
        mram_read((__mram_ptr void const*) (mram_base_addr_Y + byte_index), cache_Y, l_size_bytes);

        // Computer vector addition
        //@@ INSERT CALL TO AXPY FUNCTION HERE
        axpy (cache_Y, cache_X, alpha,l_size_bytes >> DIV);

        // Write cache to current MRAM block
        //@@ INSERT WRAM-MRAM TRANSFER HERE
        mram_write(cache_Y, (__mram_ptr void*) (mram_base_addr_Y + byte_index), l_size_bytes);
        //printf("cache_Y = %d [%d] *** byte_index = %d *** l_size_bytes = %d\n",cache_Y[byte_index],tasklet_id,byte_index,l_size_bytes);
        /*if(tasklet_id == 0){
            for (unsigned int i = 0; i < (BLOCK_SIZE >> DIV); i++) {
                printf("cache_Y = %d [%d] \n",cache_Y[i],tasklet_id);
            }
        }
        */
    }
    for (unsigned int i = 0; i < (BLOCK_SIZE >> DIV); i++) {
        printf("cache_Y = %d [%d] \n",cache_Y[i],tasklet_id);
    }

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}

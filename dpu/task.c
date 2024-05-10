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
        //bufferY[i] = 1;
        //printf("bufferY[i] = %d \n",bufferY[i]);
    }

}

static void calc_L_matrix(T *bufferL, T *bufferU, T *bufferU_inv, T *bufferA, unsigned int j, unsigned int i) {
    
    //unsigned int j = tasklet_id;
    //for (int j = 0; j < size; j++) {
		//if j is smaller than i, set l[j][i] to 0
        //printf("I= %d ******* J= %d \n",i,j);
        //bufferL[j] = 10;
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
                bufferL[i] = bufferL[i] - bufferL[k] * bufferU_inv[k];
            }
        }
}

static void calc_U_matrix(T *bufferL, T *bufferU, T *bufferU_inv, T *bufferA, T *bufferA_inv, unsigned int j, unsigned int i) {
    
    //unsigned int j = tasklet_id;
    //for (int j = 0; j < size; j++) {
		//if j is smaller than i, set l[j][i] to 0
        //printf("I= %d ******* J= %d \n",i,j);
        //bufferL[j] = 10;
		if (j < i)
		{
		    bufferU_inv[i] = 0;
        }
		else if (j == i)
		{
		    bufferU_inv[i] = 1;
        }
        else
        {
            bufferU_inv[i] = bufferA_inv[i]/bufferL[i];
            for (unsigned int k = 0; k < i; k++)
            {
                //deduct from the current l cell the value of these 2 values multiplied
                bufferU_inv[i] = bufferU_inv[i] - ((bufferL[k] * bufferU_inv[k])/bufferL[i]);
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

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size; // Transfer input size per DPU in bytes
    T alpha = DPU_INPUT_ARGUMENTS.alpha; // alpha (a in axpy)
    printf("bbbbb %f \n", DPU_INPUT_ARGUMENTS.alpha);

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_U = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);
    uint32_t mram_base_addr_L = (uint32_t)(DPU_MRAM_HEAP_POINTER + (input_size_dpu_bytes_transfer*2));
    uint32_t mram_base_addr_U_inv = (uint32_t)(DPU_MRAM_HEAP_POINTER + (input_size_dpu_bytes_transfer*3));
    uint32_t mram_base_addr_A_inv = (uint32_t)(DPU_MRAM_HEAP_POINTER + (input_size_dpu_bytes_transfer*4));

    // Initialize a local cache in WRAM to store the MRAM block
    
    /*
    printf("input_size_dpu_bytes [%d] =  %d \n",tasklet_id, input_size_dpu_bytes);
    printf("input_size_dpu_bytes_transfer [%d] =  %d \n",tasklet_id, input_size_dpu_bytes_transfer);
    printf("base_tasklet [%d] =  %d \n",tasklet_id, base_tasklet);
    printf("BLOCK_SIZE_LOG2 [%d] =  %d \n",tasklet_id, BLOCK_SIZE_LOG2);
    printf("BLOCK_SIZE [%d] =  %d \n",tasklet_id, BLOCK_SIZE);
    */

    T *cache_A = (T *) mem_alloc(BLOCK_SIZE);   
    T *cache_U = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_L = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_U_inv = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_U_inv_v2 = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_L_v2 = (T *) mem_alloc(BLOCK_SIZE);
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

    
	
    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){

        // Bound checking
        //@@ INSERT BOUND CHECKING HERE
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;
        //printf("l_size_bytes = %d \n ", l_size_bytes);
        // Load cache with current MRAM block
        //@@ INSERT MRAM-WRAM TRANSFERS HERE
        mram_read((__mram_ptr void const*) (mram_base_addr_A + byte_index), cache_A, l_size_bytes); 
        mram_read((__mram_ptr void const*) (mram_base_addr_U + byte_index), cache_U, l_size_bytes);
        mram_read((__mram_ptr void const*) (mram_base_addr_L + byte_index), cache_L, l_size_bytes);
        mram_read((__mram_ptr void const*) (mram_base_addr_U_inv + byte_index), cache_U_inv, l_size_bytes);
        mram_read((__mram_ptr void const*) (mram_base_addr_A_inv + byte_index), cache_A_inv, l_size_bytes);

        // Computer vector addition
        //@@ INSERT CALL TO AXPY FUNCTION HERE
        //axpy (cache_L, cache_U, alpha,l_size_bytes >> DIV); 
        unsigned int j = tasklet_id;
        for(unsigned int i = 0; i<8;i++){
            mram_read((__mram_ptr void const*) (mram_base_addr_U_inv + (l_size_bytes*i)), cache_U_inv_v2, l_size_bytes);
            
            calc_L_matrix(cache_L, cache_U, cache_U_inv_v2, cache_A, j, i);
            mram_write(cache_L, (__mram_ptr void*) (mram_base_addr_L + byte_index), l_size_bytes);
            
            mram_read((__mram_ptr void const*) (mram_base_addr_L + byte_index), cache_L, l_size_bytes);
            
            barrier_wait(&my_barrier);

            mram_read((__mram_ptr void const*) (mram_base_addr_L + (l_size_bytes*i)), cache_L_v2, l_size_bytes);
            printf("j= %d **** cache_L_v2 = %f \n", j,cache_L_v2[i]);
            calc_U_matrix(cache_L_v2, cache_U, cache_U_inv, cache_A, cache_A_inv, j, i);

            mram_write(cache_U_inv, (__mram_ptr void*) (mram_base_addr_U_inv + byte_index), l_size_bytes);
            mram_read((__mram_ptr void const*) (mram_base_addr_U_inv + byte_index), cache_U_inv, l_size_bytes);

            barrier_wait(&my_barrier);
            //mram_write(cache_U_inv, (__mram_ptr void*) (mram_base_addr_U_inv + byte_index), l_size_bytes);
        //barrier_wait(&my_barrier);
        }
            /*if (j < i)
            {
                //cache_L[(j*8)+i] = j;
                cache_L[2] = 10;
            }*/
            /*else
            {
                cache_L[(j*8)+i] = cache_A[(j*8)+i];
                for (unsigned int k = 0; k < i; k++)
                {
                    //deduct from the current l cell the value of these 2 values multiplied
                    cache_L[(j*8)+i] = cache_L[(j*8)+i] - cache_L[j*8+k] * cache_U[k+i*8];
                }
            }*/
        
        printf("byte_index == %d \n",byte_index);
        printf("input_size_dpu_bytes == %d \n",input_size_dpu_bytes);
    



        // Write cache to current MRAM block
        //@@ INSERT WRAM-MRAM TRANSFER HERE
        mram_write(cache_L, (__mram_ptr void*) (mram_base_addr_L + byte_index), l_size_bytes);
        mram_write(cache_U_inv, (__mram_ptr void*) (mram_base_addr_U_inv + byte_index), l_size_bytes);
        //printf("cache_Y = %d [%d] *** byte_index = %d *** l_size_bytes = %d\n",cache_Y[byte_index],tasklet_id,byte_index,l_size_bytes);
        /*if(tasklet_id == 0){
            for (unsigned int i = 0; i < (BLOCK_SIZE >> DIV); i++) {
                printf("cache_Y = %d [%d] \n",cache_Y[i],tasklet_id);
            }
        }
        */
    }
    //for (unsigned int i = 0; i < (BLOCK_SIZE >> DIV); i++) {
    //    printf("cache_Y = %f [%d] \n",cache_L[i],tasklet_id);
    //}
    /*
    for (int i = 0; i < 8; ++i){
		for (int i_aux = 0; i_aux < 8; ++i_aux){
			float aux_v2 = 0;
			for (int j = 0; j < 8; ++j){
				aux_v2 = aux_v2 + L_matrix[i*8 + j]*U_matrix[j*8 + i_aux];
                
			}
            printf("%f ", aux_v2);
			//a_aux[i][i_aux] = aux_v2;
		}
        printf("\n");
	}  
    */

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}

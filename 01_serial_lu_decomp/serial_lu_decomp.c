#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

void print_matrix_2D(float *matrix1, int input_size, int input_line_size, int lu_multiply, bool *status){
    
    if (lu_multiply == 0){
        for ( int i = 0; i < input_size; i++) {
            printf("%f ",matrix1[i]);
            if((i+1)%input_line_size == 0){
                printf("\n");
            }
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
static void read_input( float* A, const char *filename, int nr_elements){
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

void l_u_d(float *a, float *l, float *u, int size, int size_line)
{
	//for each column...
	for (int i = 0; i < size_line; i++)
	{
		//for each row....
		for (int j = 0; j < size_line; j++)
		{
			//if j is smaller than i, set l[j][i] to 0
			if (j < i)
			{
				l[(j*size_line) + i] = 0; //[j][i]
                continue;
			}
			//otherwise, do some math to get the right value
			l[(j*size_line) + i] = a[(j*size_line) + i];
			for (int k = 0; k < i; k++)
			{
				//deduct from the current l cell the value of these 2 values multiplied
				l[(j*size_line) + i] = l[(j*size_line) + i] - l[(j*size_line) + k] * u[(k*size_line) + i];
			}
		}
		//for each row...
		for (int j = 0; j < size_line; j++)
		{
			//if j is smaller than i, set u's current index to 0
			if (j < i)
			{
				u[(i*size_line) + j] = 0;
                continue;
			}
			//if they're equal, set u's current index to 1
			if (j == i)
			{
				u[(i*size_line) + j] = 1;
                continue;
			}
			//otherwise, do some math to get the right value
			u[(i*size_line) + j] = a[(i*size_line) + j] / l[(size_line + 1)*i];
			for (int k = 0; k < i; k++)
			{
				u[(i*size_line) + j] = u[(i*size_line) + j] - ((l[(i*size_line) + k] * u[(k*size_line) + j]) / l[(size_line + 1)*i]);
			}
			
		}
	}
}

int main() {

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
    

    float *A_matrix = (float *)malloc(input_size * 4);
    float *U_matrix = (float *)malloc(input_size * 4);
    float *L_matrix = (float *)malloc(input_size * 4);
    float *U_init_inv = (float *)malloc(input_size * 4);

    read_input(A_matrix, filename, input_size);

    clock_t t; 
    t = clock(); 

    l_u_d(A_matrix, L_matrix, U_matrix, input_size,input_line_size);

    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 

    // Check output
    bool status = true;
    printf(" \n ********* Lower Matrix ******************** \n");
    
    print_matrix_2D(L_matrix, input_size, input_line_size,0,&status);                                    // Print L matrix

    printf("********* Lower Matrix ******************** \n");

    printf("\n ********* Upper Matrix ******************** \n");
    
    print_matrix_2D(U_matrix, input_size, input_line_size,0,&status);                                 // Print U matrix
    
    printf("********* Upper Matrix ******************** \n");   

    printf(" \n ********* Original A Matrix ******************** \n");
    
    print_matrix_2D(A_matrix, input_size, input_line_size,0,&status);                                    // Print A matrix
    
    printf("********* Original A Matrix ******************** \n");

    printf("\n ********* L*U Matrix ******************** \n");
    
    for (unsigned int i = 0; i < input_line_size; ++i){
        for (unsigned int i_aux = 0; i_aux < input_line_size; ++i_aux){
            float aux_v2 = 0;
            for (unsigned int j = 0; j < input_line_size; ++j){
                    
                aux_v2 = aux_v2 + L_matrix[i*input_line_size + j]*U_matrix[j*input_line_size + i_aux];
            }
            printf("%f ", aux_v2);
            if (fabs(A_matrix[i*input_line_size + i_aux] - aux_v2) > 0.1){
                status = false;
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

    printf("fun() took %f ms to execute \n", time_taken*1000); 

    // Deallocation
    free(A_matrix);
    free(U_matrix);
    free(L_matrix);
	
    return 0;
}

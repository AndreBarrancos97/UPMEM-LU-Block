#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define line 4096
//int size = line * line;
//float a[size];

void saveMatrix(const char *filename, int size, float *matrix, int line) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Unable to open file for writing");
        return;
    }

	fprintf(file, "%i ", line);
    // Write the dimensions
    for (int i = 0; i < size; i++) {

    	fprintf(file, "%f ", matrix[i]);
        
    }

	fprintf(file, "\n");
    fclose(file);
}

//print the matrix out
void print_matrix_2D(float *matrix, int size, int line)
{
	int j = 0;
	for (int i = 0; i < size; i++){

		if ((i%line) == 0){
			//printf("\n");
			j=0;
		}

		//printf("%f ",matrix[i]);
		j++;
	}
}

//fill the array with random values (done for a)
void random_fill(float *matrix, int size, int line)
{
	int j = 0;
	float ao;
	//fill a with random values
	for (int i = 0; i < size; i++){
		ao = ((rand()%10)+1);
		matrix[i] = ao;
		//printf("%f \n",ao);
	}

	//Ensure the matrix is diagonal dominant to guarantee invertible-ness
	//diagCount well help keep track of which column the diagonal is in
	int diagCount = 0;
    float sum = 0;
	for (int i = 0; i < size; i++){

		sum += abs(matrix[i]);

		if (((i+1+line)%line) == 0){
			//printf("%d \n", i);
			sum -= abs(matrix[diagCount]);
			matrix[diagCount] = sum + ((rand()%10)+1);
			diagCount = diagCount + (line + 1);
			sum = 0;
		}

	}
}

int main(int argc, char** argv)
{
	//float *a;

    //printf("Enter the number of elements: ");
    //scanf("%d", &n);
	int line = 4096;
	int size = line * line;

	float *a = (float *)malloc(size * 4);
	if (a == NULL) {
			fprintf(stderr, "Memory allocation failed\n");
			return 1;
		}
    //a = (float*) malloc(size * 4);

	//fill a with random values
	random_fill(a,size,line);

	printf("****A*****");
	print_matrix_2D(a, size, line);
	printf("\n ****A***** \n");

    saveMatrix("matrix_4096x4096_16777216.txt",size, a, line);

	free(a);
	return 0;
}


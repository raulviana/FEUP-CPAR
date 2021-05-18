//#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <papi.h>
#include <chrono>

using namespace std;

#define MAX_SIZE = 8192;

#define SYSTEMTIME clock_t
#define FILENAME "exer3.csv"


bool matrix_writer(string filename, double matrixToWrite[], int size){

    cout << filename;
    ofstream matrixFile;
    matrixFile.open(filename);
    if (!matrixFile.is_open()) {
        cout << "rip openning file";
        return false;
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrixFile << matrixToWrite[i*size + j] << " ";
        }
        matrixFile << endl;
    }
    matrixFile.close();
    return true;
    
}

bool matrix_generator(double matrix[], int size)
{
    for (int line = 0; line < size; line++)
    {
        for (int col = 0; col < size; col++)
        {
            
            matrix[line*size + col] = (double)(rand() % 100) + 1;
        }
    }
    cout << "matrix generated with size" << size << "x" << size << endl;
    return matrix_writer("original_matrix.csv", matrix, size);
}

bool lu_seq(double matrix[], int size){
    for (int line = 0; line < size; line++)
    {
        for (int col = line + 1; col < size; col++)
        {
            double mul = matrix[size*col + line] / matrix[size*line + line];
            for (int i = 0; i < size; i++)
            {
                matrix[col*size + i] -= mul*matrix[size*line + col];
            }

            matrix[size*col + line] = mul;
            
        }
        
    }
    
}

bool lu_splitter(double matrix[], int size){

    double* l = new double[size * size];
    double* u = new double[size * size];


    for (int line = 0; line < size; line++)
    {
        for (int col = 0; col < size; col++)
        {
            if (line > col)
            {
                l[line*size + col] = 0;
                u[line*size + col] = matrix[size*line + col];
            }
            else if (line < col)
            {
                l[line*size + col] = matrix[size*line + col];
                u[line*size + col] = 0;
            }
            else
            {
                l[line*size + col] = 1;
                u[line*size + col] = matrix[size*line + col];
            }
        }  
    }
    return true;
}


double sequential(int size){
	SYSTEMTIME Time1, Time2;

    double* matrix = new double[size * size];
    matrix_generator(matrix, size);
	auto start = std::chrono::steady_clock::now();
	  lu_seq(matrix, size);
    lu_splitter(matrix, size);
  
	auto end = std::chrono::steady_clock::now();

	return (double)(end - start).count();
}

void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}

void init_papi()
{
	int retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT && retval < 0)
	{
		printf("PAPI library version mismatch!\n");
		exit(1);
	}
	if (retval < 0)
		handle_error(retval);

	std::cout << "PAPI Version Number: MAJOR: " << PAPI_VERSION_MAJOR(retval)
			  << " MINOR: " << PAPI_VERSION_MINOR(retval)
			  << " REVISION: " << PAPI_VERSION_REVISION(retval) << "\n";
}


int main(int argc, char *argv[])
{
    int size;
    srand(time(NULL));
	int EventSet = PAPI_NULL;

    int ret;
	double time;
	int op;
	ofstream fileStream;
	int startSize, endSize, step;
	int lin, col, nt = 1;
	long long values[2];
	float flops;
	
	ret = PAPI_library_init(PAPI_VER_CURRENT);
	if (ret != PAPI_VER_CURRENT)
		std::cout << "FAIL" << endl;

	ret = PAPI_create_eventset(&EventSet);
	if (ret != PAPI_OK)
		cout << "ERRO: create eventset" << endl;

	ret = PAPI_add_event(EventSet, PAPI_L1_DCM);
	if (ret != PAPI_OK)
		cout << "ERRO: PAPI_L1_DCM" << endl;

	ret = PAPI_add_event(EventSet, PAPI_L2_DCM);
	if (ret != PAPI_OK)
		cout << "ERRO: PAPI_L2_DCM" << endl;

    cout << "Enter matrix size: ";

    cin >> size;


    	op = 1;
	do
	{
		cout << endl
			 << "1. Sequential LU Decomposition" << endl;
		cout << "2. Block LU Decomposition" << endl;
		cout << "Selection?: [0 to stop] ";
		cin >> op;
		if (op == 0)
		{
			cout << "Exiting" << endl;
			break;
		}


		printf("Parameters?: [StartSize] [Step] [EndSize]\n ");
		cin >> startSize >> step >> endSize;

		int blockSize;
		if (op == 2){
			
			cout << "\n****Block Multiplication****" << endl;
			cout << "Block size?: [0 < Block Size <= " << startSize << " ]" << endl;
			cin >> blockSize;
			if (blockSize <= 0 || blockSize > startSize)
				exit(8);
		}

		fileStream.open(FILENAME, fstream::app);
		do
		{
			// Start PAPI counting
			ret = PAPI_start(EventSet);
			if (ret != PAPI_OK)
				cout << "ERRO: Start PAPI" << endl;
			lin = startSize;
			col = startSize;
			switch (op)
			{
			case 1:
				time = sequential(lin);
    		flops = (2.0f * lin * lin * lin / time) * 1.0e-9f;
				cout << "GFLOPS: " << flops;
				break;
			case 2:
				//time = OnMultLine(lin, col);
				break;
            }
			ret = PAPI_stop(EventSet, values);
			if (ret != PAPI_OK)
				cout << "ERRO: Stop PAPI" << endl;
			printf("L1 DCM: %lld \n", values[0]);
			printf("L2 DCM: %lld \n", values[1]);

			fileStream << startSize << ", " << time << ", " << flops << ", " << values[0] << ", " << values[1] << "\n";
			ret = PAPI_reset(EventSet);
			if (ret != PAPI_OK)
				std::cout << "FAIL reset" << endl;

			printf("\nCurrent size: %d\n", startSize);
			printf("*****************************\n\n");

			startSize += step;

		} while (startSize <= endSize);
		fileStream.close();

	} while (op != 0);

    	ret = PAPI_remove_event(EventSet, PAPI_L1_DCM);
	if (ret != PAPI_OK)
		std::cout << "FAIL remove event" << endl;

	ret = PAPI_remove_event(EventSet, PAPI_L2_DCM);
	if (ret != PAPI_OK)
		std::cout << "FAIL remove event" << endl;

	ret = PAPI_destroy_eventset(&EventSet);
	if (ret != PAPI_OK)
		std::cout << "FAIL destroy" << endl;
    return 1;
}
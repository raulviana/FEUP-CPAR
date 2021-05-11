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

using namespace std;

#define MAX_SIZE = 6000;


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
    return (matrix_writer("l_matrix.csv", l, size) && matrix_writer("u_matrix.csv", u, size) && matrix_writer("result_matrix.csv", matrix, size));
}

int main(int argc, char *argv[])
{
    int size;
    srand(time(NULL));

    cout << "Enter matrix size: ";

    cin >> size;

    double* matrix = new double[size * size];
    matrix_generator(matrix, size);
    
    lu_seq(matrix, size);

    lu_splitter(matrix, size);
    return 1;
}
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
    
}

bool matrix_generator(double matrix[], int size)
{
    cout << size;
    
    cout << "alloc success";
    for (int line = 0; line < size; line++)
    {
        for (int col = 0; col < size; col++)
        {
            
            matrix[line*size + col] = (double)(rand() % 11);
            cout << matrix[line*size +col];
        }
    }
    cout << "here";
    matrix_writer("abc.txt", matrix, size);

    return false;
}

int main(int argc, char *argv[])
{
    int size;
    srand(time(NULL));

    cout << "Enter matrix size: ";

    cin >> size;

    double matrix[size * size];
    matrix_generator(matrix, size);
    return 1;
}
/*  This example compares an OpenMP blocked matrix multiplication
 *  implementation with a SYCL blocked matrix multiplication example.
 *  The purpose is not to compare performance, but to show the similarities
 *  and differences between them.
 *  See block_host for the OpenMP implementation. */

#include <omp.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>

#define FILENAME "openMP.csv"

using namespace std;

class mxm_kernel;

void display_matrix(float *m, int matSize)
{
    std::cout << std::endl
              << "=================" << std::endl;
    for (int i = 0; i < matSize; i++)
    {
        std::cout << m[i] << " ";
        if (i > 10)
            break;
    }
    std::cout << std::endl
              << "=================" << std::endl;
}

inline int prevPowerOfTwo(int x)
{
    if (x < 2)
    {
        return 0;
    }
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x - (x >> 1);
}

/* Function template that performs the matrix * matrix operation. It is
 * a template because only some OpenCL devices support double-precision
 * floating-point numbers.
 * Broadly, the function chooses an appropriate work size, then enqueues
 * the matrix * matrix lambda on the queue provided.
 * Note that this example only works for powers of two.
 * */
template <typename T>
void local_mxm(T *MA, T *MB, T *MC, unsigned int matSize)
{
    std::ofstream fileStream;

    // Make sure the block size is not larger than the mat size
    blockSize = std::min(matSize, blockSize);
    auto start = std::chrono::steady_clock::now();

    for (int kk = 0; kk < m_ar; kk += blockSize)
    {
        for (int jj = 0; jj < m_ar; jj += blockSize)
        {
            for (int i = 0; i < m_ar; i++)
            {
                int k = kk;
#pragma omp parallel for collapse(2)
                for (k; k < (kk + blockSize); k++)
                {
                    int j = jj;
                    for (j; j < (jj + blockSize); j++)
                    {
#pragma omp critical
                        MC[i * m_ar + j] += MA[i * m_ar + k] * MB[k * m_ar + j];
                    }
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_seconds = end - start;
    std::cout << "Time : " << time_seconds.count() << " seconds" << std::endl;
    float flops = (2.0f * matSize * matSize * matSize / time_seconds.count()) * 1.0e-9f;
    std::cout << "GFLOPs: " << flops << std::endl;
    fileStream.open(FILENAME, fstream::app);
    fileStream << matSize << ", " << time_seconds.count() << ", " << flops << "\n";
    fileStream.close();
    display_matrix(MC, matSize);
}

int main(int argc, char *argv[])
{
    float *MA;
    float *MB;
    float *MC;

    bool error = false;

    int op;
    unsigned int startSize, endSize, step;
    int platformIndex = 1;
    int deviceIndex = 1;

    std::cout << std::endl
              << "********** openMP Matrix Multiplication ****************" << std::endl
              << std::endl;
    op = 1;
    do
    {
        std::cout << "Selection?: 1 => Block Multiplication | 0 => Stop" << std::endl;
        std::cin >> op;
        if (op == 0)
        {
            std::cout << "Exiting" << std::endl;
            return 0;
        }

        printf("Parameters?: [StartSize] [Step] [EndSize]\n ");
        std::cin >> startSize >> step >> endSize;

        int blockSize;

        std::cout << "\n****Parallel Block Multiplication****" << std::endl;

        do
        {
            MA = new float[startSize * startSize];
            MB = new float[startSize * startSize];
            MC = new float[startSize * startSize];

            for (int i = 0; i < startSize; i++)
                for (int j = 0; j < startSize; j++)
                    MA[i * startSize + j] = (double)1.0;

            for (int i = 0; i < startSize; i++)
                for (int j = 0; j < startSize; j++)
                    MB[i * startSize + j] = (double)(i + 1);

            //call the execution function
            local_mxm(MA, MB, MC, startSize);
            startSize += step;

            delete[] MA;
            delete[] MB;
            delete[] MC;
        } while (startSize <= endSize);

    } while (op != 0);

    return 0;
}

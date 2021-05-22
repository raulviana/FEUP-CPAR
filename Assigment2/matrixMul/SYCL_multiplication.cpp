
#include <CL/sycl.hpp>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>

#define FILENAME "assingment2.csv"

using namespace cl::sycl;
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


template <typename T>
void local_mxm(cl::sycl::queue &q, T *MA, T *MB, T *MC, unsigned int matSize)
{
    std::ofstream fileStream;
    auto device = q.get_device();
    auto maxBlockSize = device.get_info<cl::sycl::info::device::max_work_group_size>();
    unsigned int blockSize = prevPowerOfTwo(std::sqrt(maxBlockSize));
    std::cout << "using device : " << device.get_info<cl::sycl::info::device::name>();
    std::cout << " The Device Max Work Group Size is : " << maxBlockSize << std::endl;
    std::cout << " The matrix size is : " << matSize << std::endl;
    std::cout << " The maximume blockSize is : " << blockSize << std::endl;

    // Make sure the block size is not larger than the mat size
    blockSize = std::min(matSize, blockSize);
    auto start = std::chrono::steady_clock::now();
    {
        range<1> dimensions(matSize * matSize);
        const property_list props = {property::buffer::use_host_ptr()};
        buffer<T> bA(MA, dimensions, props);
        buffer<T> bB(MB, dimensions, props);
        buffer<T> bC(MC, dimensions, props);

        q.submit([&](handler &cgh) {
            auto pA = bA.template get_access<access::mode::read>(cgh);
            auto pB = bB.template get_access<access::mode::read>(cgh);
            auto pC = bC.template get_access<access::mode::write>(cgh);
            auto localRange = range<1>(blockSize * blockSize);

            accessor<T, 1, access::mode::read_write, access::target::local> pBA(
                localRange, cgh);
            accessor<T, 1, access::mode::read_write, access::target::local> pBB(
                localRange, cgh);

            cgh.parallel_for<mxm_kernel>(
                nd_range<2>{range<2>(matSize, matSize), range<2>(blockSize, blockSize)}, [=](nd_item<2> it) {
                    // Current block
                    int blockX = it.get_group(0);
                    int blockY = it.get_group(1);

                    // Current local item
                    int localX = it.get_local_id(0);
                    int localY = it.get_local_id(1);

                    // Start in the A matrix
                    int a_start = matSize * blockSize * blockY;
                    // End in the b matrix
                    int a_end = a_start + matSize - 1;
                    // Start in the b matrix
                    int b_start = blockSize * blockX;

                    // Result for the current C(i,j) element
                    T tmp = 0.0f;
                    // We go through all a, b blocks
                    for (int a = a_start, b = b_start; a <= a_end;
                         a += blockSize, b += (blockSize * matSize))
                    {
                        // Copy the values in shared memory collectively
                        pBA[localY * blockSize + localX] =
                            pA[a + matSize * localY + localX];
                        // Note the swap of X/Y to maintain contiguous access
                        pBB[localX * blockSize + localY] =
                            pB[b + matSize * localY + localX];
                        it.barrier(access::fence_space::local_space);
                        // Now each thread adds the value of its sum
                        for (int k = 0; k < blockSize; k++)
                        {
                            tmp +=
                                pBA[localY * blockSize + k] * pBB[localX * blockSize + k];
                        }
                        // The barrier ensures that all threads have written to local
                        // memory before continuing
                        it.barrier(access::fence_space::local_space);
                    }
                    auto elemIndex = it.get_global_id(1) * it.get_global_range()[0] +
                                     it.get_global_id(0);
                    // Each thread updates its position
                    pC[elemIndex] = tmp;
                });
        });
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
              << "********** SYCL Matrix Multiplication ****************" << std::endl
              << std::endl;
    op = 1;
    do
    {
        platformIndex = 1;
        deviceIndex = 1;

        auto platforms = sycl::platform::get_platforms();
        std::cout << std::endl
                  << std::endl;
        for (auto &platform : platforms)
        {
            std::cout << platformIndex << " Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

            auto devices = platform.get_devices();
            for (auto &device : devices)
            {
                std::cout << "  " << deviceIndex << " Device: " << device.get_info<sycl::info::device::name>() << std::endl;
                deviceIndex++;
            }
            deviceIndex = 1;
            platformIndex++;
        }
        std::cout << "Please select desired [Platform] [Device]:" << std::endl;
        std::cin >> platformIndex >> deviceIndex;
        auto devices = platforms[platformIndex - 1].get_devices();
        device d;
        d = device(devices[deviceIndex - 1]);

        std::cout << "Using " << d.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "max_work_group: " << d.get_info<sycl::info::device::max_work_group_size>() << std::endl;
        std::cout << "max work item sizes: " << d.get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
        std::cout << "number cores: " << d.get_info<sycl::info::device::max_compute_units>() << std::endl;

        queue q(d);
        std::cout << std::endl;
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
            local_mxm(q, MA, MB, MC, startSize);
            startSize += step;

            delete[] MA;
            delete[] MB;
            delete[] MC;
        } while (startSize <= endSize);

    } while (op != 0);

    return 0;
}

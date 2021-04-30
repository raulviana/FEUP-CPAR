#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

using namespace cl::sycl;

class mxm_kernel;

void display_matrix(float* m, int matSize) {
if (matSize > 16) {
    return;
}

std::cout << "=======" << std::endl;
for (int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
        std::cout << m[i * matSize + j] << " ";
    }
    std::cout << std::endl;
}
std::cout << "=======" << std::endl;
;
}

inline int prevPowerOfTwo(int x) {
if (x < 0) {
    return 0;
}
--x;
x |= x >> 1;
x |= x >> 2;
x |= x >> 4;
x |= x >> 8;
x |= x >> 16;
return x - (x >> 1);
}

inline bool isPowerOfTwo(int x) {
return (x & (x - 1)) == 0;
}

template <typename T>
bool local_mxm(cl::sycl::queue& q, T* MA, T* MB, T* MC, T* MD, int matSize) {
// Make sure it is power of two before running
if (!isPowerOfTwo(matSize)) {
    std::cout << " This example only works with power of two sizes "
        << std::endl;
    return true;
}

auto device = q.get_device();
auto maxBlockSize =
    device.get_info<cl::sycl::info::device::max_work_group_size>();
auto blockSize = prevPowerOfTwo(std::sqrt(maxBlockSize));
std::cout << " The Device Max Work Group Size is : " << maxBlockSize
    << std::endl;
std::cout << " The order is : " << matSize << std::endl;
std::cout << " The blockSize is : " << blockSize << std::endl;
// Make sure the block size is not larger than the mat size
blockSize = std::min(matSize, blockSize);

{
    range<1> dimensions(matSize * matSize);
    const property_list props = { property::buffer::use_host_ptr() };
    buffer<T> bA(MA, dimensions, props);
    buffer<T> bB(MB, dimensions, props);
    buffer<T> bC(MC, dimensions, props);
    buffer<T> bD(MD, dimensions, props);

    q.submit([&](handler& cgh) {
        auto pA = bA.template get_access<access::mode::read>(cgh);
        auto pB = bB.template get_access<access::mode::read>(cgh);
        auto pC = bC.template get_access<access::mode::write>(cgh);
        auto pD = bD.template get_access<access::mode::write>(cgh);
        auto localRange = range<1>(blockSize * blockSize);

        accessor<T, 1, access::mode::read_write, access::target::local> pBA(
            localRange, cgh);
        accessor<T, 1, access::mode::read_write, access::target::local> pBB(
            localRange, cgh);

        cgh.parallel_for<class matrix_add>(range<2> {matSize, matSize}, [=](id<2> it) {
            pD[it] = pA[it] + pB[it];
        });

        cgh.parallel_for<mxm_kernel>(
            nd_range<2>{range<2>(matSize, matSize),
            range<2>(blockSize, blockSize)},
            [=](nd_item<2> it) {
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
                a += blockSize, b += (blockSize * matSize)) {
                // Copy the values in shared memory collectively
                pBA[localY * blockSize + localX] =
                    pA[a + matSize * localY + localX];
                // Note the swap of X/Y to maintain contiguous access
                pBB[localX * blockSize + localY] =
                    pB[b + matSize * localY + localX];
                it.barrier(access::fence_space::local_space);
                // Now each thread adds the value of its sum
                for (int k = 0; k < blockSize; k++) {
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
return false;
}

int main(int argc, char* argv[]) {
float* MA;
float* MB;
float* MC;
float* MD;
bool sycl = true;
bool error = false;

int matSize = 4;
MA = new float[matSize * matSize];
MB = new float[matSize * matSize];
MC = new float[matSize * matSize];
MD = new float[matSize * matSize];



std::cout << " Input matrix " << std::endl;
display_matrix(MA, matSize);
display_matrix(MB, matSize);
display_matrix(MC, matSize);
display_matrix(MD, matSize);



if (sycl) {
    std::cout << " ***** SYCL " << std::endl;
    // MatrixC initialization
    std::cout << "MATRIX D" << std::endl;
    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            MD[i * matSize + j] = 0.0f;  // i * matSize + j;
            std::cout << MD[i * matSize + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======" << std::endl;
    // MatrixC initialization
    std::cout << "MATRIX C" << std::endl;
    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            MC[i * matSize + j] = 0.0f;  // i * matSize + j;
            std::cout << MC[i * matSize + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======" << std::endl;
    // MatrixA initialization
    std::cout << "MATRIX A" << std::endl;
    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            MA[i * matSize + j] = 0.0f + j;  // i * matSize + j;
            std::cout << MA[i * matSize + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======" << std::endl;
    // MatrixB initialization
    std::cout << "MATRIX B" << std::endl;
    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            MB[i * matSize + j] = 0.0f + j;  // i * matSize + j;
            std::cout << MB[i * matSize + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======" << std::endl;
    {
        {
            cpu_selector device_selector;
            queue q(device_selector);


            auto start = std::chrono::steady_clock::now();
            error = local_mxm(q, MA, MB, MC, MD, matSize);
            q.wait_and_throw();
            auto end = std::chrono::steady_clock::now();
            auto time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
            std::cout << "SYCL: ";
            std::cout << "Time: " << time << std::endl;
            float flops =
                (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
            std::cout << "GFLOPs: " << flops << std::endl;
            std::cout << " Output " << std::endl;
        }
        display_matrix(MC, matSize);
        display_matrix(MD, matSize);
    }
}

delete[] MA;
delete[] MB;
delete[] MC;

return error ? 1 : 0;
}
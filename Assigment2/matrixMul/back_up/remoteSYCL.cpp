/*  This example compares an OpenMP blocked matrix multiplication
 *  implementation with a SYCL blocked matrix multiplication example.
 *  The purpose is not to compare performance, but to show the similarities
 *  and differences between them.
 *  See block_host for the OpenMP implementation. */

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

void display_matrix(float* m, int matSize) {
  if (matSize > 11) {
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

/* Function template that performs the matrix * matrix operation. (It is
 * a template because only some OpenCL devices support double-precision
 * floating-point numbers, but it is interesting to make the comparison
 * where available.)
 * Broadly, the function chooses an appropriate work size, then enqueues
 * the matrix * matrix lambda on the queue provided. Because the queues
 * are constructed inside this function, it will block until the work is
 * finished.
 * Note that this example only works for powers of two.
 * */
template <typename T>
void compute(int matSize, int blockSize) {
//   float* MA;
// float* MB;
// float* MC;
// float* MD;
// MA = new float[matSize * matSize];
// MB = new float[matSize * matSize];
// MC = new float[matSize * matSize];
// MD = new float[matSize * matSize];

//    std::cout << " ***** SYCL " << std::endl;
//     // MatrixC initialization
//     std::cout << "MATRIX D" << std::endl;
//     for (int i = 0; i < matSize; i++) {
//         for (int j = 0; j < matSize; j++) {
//             MD[i * matSize + j] = 0.0f;  // i * matSize + j;
//             std::cout << MD[i * matSize + j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "=======" << std::endl;
//     // MatrixC initialization
//     std::cout << "MATRIX C" << std::endl;
//     for (int i = 0; i < matSize; i++) {
//         for (int j = 0; j < matSize; j++) {
//             MC[i * matSize + j] = 0.0f;  // i * matSize + j;
//             std::cout << MC[i * matSize + j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "=======" << std::endl;
//     // MatrixA initialization
//     std::cout << "MATRIX A" << std::endl;
//     for (int i = 0; i < matSize; i++) {
//         for (int j = 0; j < matSize; j++) {
//             MA[i * matSize + j] = 0.0f + j;  // i * matSize + j;
//             std::cout << MA[i * matSize + j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "=======" << std::endl;
//     // MatrixB initialization
//     std::cout << "MATRIX B" << std::endl;
//     for (int i = 0; i < matSize; i++) {
//         for (int j = 0; j < matSize; j++) {
//             MB[i * matSize + j] = 0.0f + j;  // i * matSize + j;
//             std::cout << MB[i * matSize + j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "=======" << std::endl;
//     {
//         {
//             cpu_selector device_selector;
//             queue q(device_selector);


//             auto start = std::chrono::steady_clock::now();


// auto device = q.get_device();
// auto maxBlockSize =
//     device.get_info<cl::sycl::info::device::max_work_group_size>();
// auto blockSize = prevPowerOfTwo(std::sqrt(maxBlockSize));
// std::cout << " The Device Max Work Group Size is : " << maxBlockSize
//     << std::endl;
// std::cout << " The order is : " << matSize << std::endl;
// std::cout << " The blockSize is : " << blockSize << std::endl;
// // Make sure the block size is not larger than the mat size
// blockSize = std::min(matSize, blockSize);

// {
//     range<1> dimensions(matSize * matSize);
//     const property_list props = { property::buffer::use_host_ptr() };
//     buffer<T> bA(MA, dimensions, props);
//     buffer<T> bB(MB, dimensions, props);
//     buffer<T> bC(MC, dimensions, props);
//     buffer<T> bD(MD, dimensions, props);

//     q.submit([&](handler& cgh) {
//         auto pA = bA.template get_access<access::mode::read>(cgh);
//         auto pB = bB.template get_access<access::mode::read>(cgh);
//         auto pC = bC.template get_access<access::mode::write>(cgh);
//         auto pD = bD.template get_access<access::mode::write>(cgh);
//         auto localRange = range<1>(blockSize * blockSize);

//         accessor<T, 1, access::mode::read_write, access::target::local> pBA(
//             localRange, cgh);
//         accessor<T, 1, access::mode::read_write, access::target::local> pBB(
//             localRange, cgh);

//         cgh.parallel_for<class matrix_add>(range<2> {matSize, matSize}, [=](id<2> it) {
//             pD[it] = pA[it] + pB[it];
//         });

//         cgh.parallel_for<mxm_kernel>(
//             nd_range<2>{range<2>(matSize, matSize),
//             range<2>(blockSize, blockSize)},
//             [=](nd_item<2> it) {
//             // Current block
//             int blockX = it.get_group(0);
//             int blockY = it.get_group(1);

//             // Current local item
//             int localX = it.get_local_id(0);
//             int localY = it.get_local_id(1);

//             // Start in the A matrix
//             int a_start = matSize * blockSize * blockY;
//             // End in the b matrix
//             int a_end = a_start + matSize - 1;
//             // Start in the b matrix
//             int b_start = blockSize * blockX;

//             // Result for the current C(i,j) element
//             T tmp = 0.0f;
//             // We go through all a, b blocks
//             for (int a = a_start, b = b_start; a <= a_end;
//                 a += blockSize, b += (blockSize * matSize)) {
//                 // Copy the values in shared memory collectively
//                 pBA[localY * blockSize + localX] =
//                     pA[a + matSize * localY + localX];
//                 // Note the swap of X/Y to maintain contiguous access
//                 pBB[localX * blockSize + localY] =
//                     pB[b + matSize * localY + localX];
//                 it.barrier(access::fence_space::local_space);
//                 // Now each thread adds the value of its sum
//                 for (int k = 0; k < blockSize; k++) {
//                     tmp +=
//                         pBA[localY * blockSize + k] * pBB[localX * blockSize + k];
//                 }
//                 // The barrier ensures that all threads have written to local
//                 // memory before continuing
//                 it.barrier(access::fence_space::local_space);
//             }
//             auto elemIndex = it.get_global_id(1) * it.get_global_range()[0] +
//                 it.get_global_id(0);
//             // Each thread updates its position
//             pC[elemIndex] = tmp;
//         });
//     });
// }
  
//     q.wait_and_throw();
//             auto end = std::chrono::steady_clock::now();
//             auto time =
//                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
//                 .count();
//             std::cout << "SYCL: ";
//             std::cout << "Time: " << time << std::endl;
//             float flops =
//                 (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
//             std::cout << "GFLOPs: " << flops << std::endl;
//             std::cout << " Output " << std::endl;
//         }
//         display_matrix(MC, matSize);
//         display_matrix(MD, matSize);
//     }
// }

// delete[] MA;
// delete[] MB;
// delete[] MC;
}


int main(int argc, char* argv[]) {
  
  int matSize;
  int op;
  int startSize, endSize, step;
  ofstream fileStream;

   auto platforms = sycl::platform::get_platforms();

  for (auto &platform : platforms) {
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
              << std::endl;

    auto devices = platform.get_devices();
    for (auto &device : devices) {
      std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                << std::endl;
    }
  }
  exit(8);

  op = 1;
  do {
    std::cout << std::endl;
    std::cout << "Selection?: 1 => Block Multiplication | 0 => Stop" << std::endl;
    std::cin >> op;
    if (op == 0) {
        std::cout << "Exiting" << std::endl;
        exit(3);
    }

    printf("Parameters?: [StartSize] [Step] [EndSize]\n ");
    std::cin >> startSize >> step >> endSize;

    int blockSize;

    std::cout << "\n****Block Multiplication****" << std::endl;
    std::cout << "Block size?: [0 < Block Size <= " << startSize << " ]" << std::endl;
    std::cin >> blockSize;
    if (blockSize <= 0 || blockSize > startSize)
        exit(8);

    fileStream.open(FILENAME, fstream::app);
    do {
      //call the execution function
      compute(startSize, blockSize);
      startSize += step;

    } while (startSize <= endSize);
    fileStream.close();

  } while (op != 0);




  return 0;
}
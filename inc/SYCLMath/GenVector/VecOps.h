#include "SYCLMath/GenVector/MathUtil.h"
#include <cstdio>
#include <chrono>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define ERRCHECK(err) __checkCudaErrors((err), __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}


// namespace ROOT {
// namespace Experimental {

namespace ROOT
{

  namespace Experimental
  {

    template <class Scalar, class LVector>
    __global__ void InvariantMassKernel(LVector* vec, Scalar* m, size_t N)
    {
      int id = blockDim.x * blockIdx.x + threadIdx.x;
      if (id < N)
      {
        LVector w = vec[id];
        m[id] = w.mass();
      }
    }

    template <class Scalar, class LVector>
    __global__ void InvariantMassesKernel(LVector* v1, LVector* v2, Scalar* m, size_t N)
    {
      int id = blockDim.x * blockIdx.x + threadIdx.x;
      if (id < N)
      {
        LVector w = v1[id] + v2[id];
        m[id] = w.mass();
      }
    }

    template <class Scalar, class LVector>
    Scalar *InvariantMasses(LVector *v1, LVector *v2, const size_t N,
                            const size_t local_size)
    {

      Scalar *invMasses = new Scalar[N];

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif

      {
        // Allocate device input vector
        LVector *d_v1 = NULL;
        ERRCHECK(cudaMalloc((void **)&d_v1, N * sizeof(LVector)));

        // Allocate device input vector
        LVector *d_v2 = NULL;
        ERRCHECK(cudaMalloc((void **)&d_v2, N * sizeof(LVector)));

        // Allocate the device output vector
        Scalar *d_invMasses = NULL;
        ERRCHECK(cudaMalloc((void **)&d_invMasses, N * sizeof(Scalar)));

        cudaMemcpy(d_v1, v1, N * sizeof(LVector), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, v2, N * sizeof(LVector), cudaMemcpyHostToDevice);

        InvariantMassesKernel<<<N / local_size + 1, local_size>>>(d_v1, d_v2, d_invMasses, N);

        cudaDeviceSynchronize();
        ERRCHECK(cudaMemcpy(invMasses, d_invMasses, N * sizeof(Scalar), cudaMemcpyDeviceToHost));

        ERRCHECK(cudaFree(d_v1));
        ERRCHECK(cudaFree(d_v2));
        ERRCHECK(cudaFree(d_invMasses));
      }

#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "cuda time " << duration << " (s)" << std::endl;
#endif

      return invMasses;
    }

    template <class Scalar, class LVector>
    Scalar *InvariantMass(LVector *v1, const size_t N, const size_t local_size)
    {

      Scalar *invMasses = new Scalar[N];

#ifdef ROOT_MEAS_TIMING
      auto start = std::chrono::system_clock::now();
#endif
      {
        // Allocate the device input vector
        LVector* d_v1 = NULL;
        ERRCHECK(cudaMalloc((void **)&d_v1, N * sizeof(LVector)));

        // Allocate the device output vector
        Scalar *d_invMasses = NULL;
        ERRCHECK(cudaMalloc((void **)&d_invMasses, N * sizeof(Scalar)));
        ERRCHECK(cudaMemcpy(d_v1, v1, N*sizeof(LVector), cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();
        InvariantMassKernel<<<N / local_size + 1, local_size>>>(d_v1, d_invMasses, N);

        ERRCHECK(cudaMemcpy(invMasses, d_invMasses, N * sizeof(Scalar), cudaMemcpyDeviceToHost));

        ERRCHECK(cudaFree(d_v1));
        ERRCHECK(cudaFree(d_invMasses));
      }

#ifdef ROOT_MEAS_TIMING
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() *
          1e-6;
      std::cout << "cuda time " << duration << " (s)" << std::endl;
#endif

      return invMasses;
    }

  } // namespace Experimental
} // namespace ROOT

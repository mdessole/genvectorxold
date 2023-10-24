#include "SYCLMath/Vector4D.h"
#include <chrono>
#include <sycl/sycl.hpp>

using arithmetic_type = double;
using vec4d = ROOT::Experimental::LorentzVector<
    ROOT::Experimental::PtEtaPhiE4D<arithmetic_type>>;
template <class T> using Vector = std::vector<T>;

#ifndef RVecSYCL_H
#define RVecSYCL_H

using mode = sycl::access::mode;

template <class T> using AccRW = sycl::accessor<T, 1, mode::read_write>;
template <class T> using AccR = sycl::accessor<T, 1, mode::read>;
template <class T> using AccW = sycl::accessor<T, 1, mode::write>;

// namespace ROOT {
// namespace Experimental {

template <typename AccVec, typename AccMass> class InvariantMassKernel {
public:
  InvariantMassKernel(AccVec acc_v, AccMass acc_m, size_t n)
      : vec_acc(acc_v), mass_acc(acc_m), N(n) {}

  void operator()(sycl::nd_item<1> item) {
    size_t id = item.get_global_id().get(0);
    if (id < N) {
      mass_acc[id] = vec_acc[id].mass();
    }
  }

private:
  AccVec vec_acc;
  AccMass mass_acc;
  size_t N;
};

//} // namespace Experimental
//} // namespace ROOT
#endif

Vector<arithmetic_type> InvariantMass(const Vector<vec4d> v1, const size_t N,
                                      const size_t local_size,
                                      sycl::queue queue) {

  Vector<arithmetic_type> invMasses(N);

  auto start = std::chrono::system_clock::now();

  std::cout << "sycl::queue check - selected device:\n"
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  { // Start of scope, ensures data copied back to host
    // Create device buffers. The memory is managed by SYCL so we should NOT
    // access these buffers directly.
    auto execution_range = sycl::nd_range<1>{
        sycl::range<1>{((N + local_size - 1) / local_size) * local_size},
        sycl::range<1>{local_size}};
    sycl::buffer<vec4d, 1> v1_sycl(v1.data(), v1.size());
    sycl::buffer<arithmetic_type, 1> im_sycl(invMasses.data(),
                                             invMasses.size());

    queue.submit([&](sycl::handler &cgh) {
      // Get handles to SYCL buffers.
      auto vec_acc = v1_sycl.get_access<mode::read>(cgh);
      auto mass_acc = im_sycl.get_access<mode::write>(cgh);

      // Partitions the vector pairs over available threads and computes the
      // invariant masses.
      cgh.parallel_for(execution_range,
                       // InvariantMassKernel<decltype(v1_acc),
                       // decltype(im_acc)>(vec_acc, mass_acc, N)
                       [=](sycl::nd_item<1> item) {
                         size_t id = item.get_global_id().get(0);
                         if (id < N) {
                           mass_acc[id] = vec_acc[id].mass();
                         }
                       });
    });
  } // end of scope, ensures data copied back to host
  queue.wait();

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "sycl time " << duration << " (s)" << std::endl;

  return invMasses;
}

Vector<arithmetic_type> InvariantMasses(const Vector<vec4d> v1,
                                        const Vector<vec4d> v2, const size_t N,
                                        const size_t local_size,
                                        sycl::queue queue) {

  Vector<arithmetic_type> invMasses(N);

  auto start = std::chrono::system_clock::now();

  std::cout << "sycl::queue check - selected device:\n"
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  { // Start of scope, ensures data copied back to host
    // Create device buffers. The memory is managed by SYCL so we should NOT
    // access these buffers directly.
    auto execution_range = sycl::nd_range<1>{
        sycl::range<1>{((N + local_size - 1) / local_size) * local_size},
        sycl::range<1>{local_size}};
    sycl::buffer<vec4d, 1> v1_sycl(v1.data(), v1.size());
    sycl::buffer<vec4d, 1> v2_sycl(v2.data(), v2.size());
    sycl::buffer<arithmetic_type, 1> im_sycl(invMasses.data(),
                                             invMasses.size());

    queue.submit([&](sycl::handler &cgh) {
      // Get handles to SYCL buffers.
      auto v1_acc = v1_sycl.get_access<mode::read>(cgh);
      auto v2_acc = v2_sycl.get_access<mode::read>(cgh);
      auto mass_acc = im_sycl.get_access<mode::write>(cgh);

      // Partitions the vector pairs over available threads and computes the
      // invariant masses.
      cgh.parallel_for(execution_range,
                       // InvariantMassKernel<decltype(v1_acc),
                       // decltype(im_acc)>(vec_acc, mass_acc, N)
                       [=](sycl::nd_item<1> item) {
                         size_t id = item.get_global_id().get(0);
                         if (id < N) {
                           auto w = v1_acc[id] + v2_acc[id];
                           mass_acc[id] = w.mass();
                         }
                       });
    });
  } // end of scope, ensures data copied back to host
  queue.wait();

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "sycl time " << duration << " (s)" << std::endl;

  return invMasses;
}

Vector<vec4d> GenVectors(int n) {
  Vector<vec4d> vectors(n);

  // generate n -4 momentum quantities
  for (int i = 0; i < n; ++i) {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

int main(int argc, char **argv) {

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  size_t local_size = 128;

  auto u_vectors = GenVectors(N);
  auto v_vectors = GenVectors(N);

  static sycl::queue queue{sycl::default_selector{}};

  Vector<arithmetic_type> masses =
      InvariantMasses(u_vectors, v_vectors, N, local_size, queue);

  assert((std::abs(masses[0] - (-2.3504)) <= 1e-5));
  return 0;
}

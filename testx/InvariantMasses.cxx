#include "ROOT/RVec.hxx"
#include "TROOT.h"
#include "Math/PtEtaPhiM4D.h"
#include "Math/Vector4D.h"
#include <assert.h>
#include <chrono>
#include <vector>

#ifdef SINGLE_PRECISION
using arithmetic_type = float;
#else
using arithmetic_type = double;
#endif

using vec4d =
    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<arithmetic_type>>;

template <class T>
using Vector = ROOT::RVec<T>;

Vector<arithmetic_type> InvariantMasses(const Vector<vec4d> v1,
                                        const Vector<vec4d> v2, const size_t N,
                                        const size_t local_size)
{

  Vector<arithmetic_type> invMasses(N);

  auto start = std::chrono::system_clock::now();

  for (size_t i = 0; i < N; i++)
  {
    auto w = v1[i] + v2[i];
    invMasses[i] = w.mass();
  }

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  // std::cout << "cpu time " << duration << " (s)" << std::endl;

  return invMasses;
}

Vector<vec4d> GenVectors(int n)
{
  Vector<vec4d> vectors(n);

  // generate n -4 momentum quantities
  for (int i = 0; i < n; ++i)
  {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

int main(int argc, char **argv)
{

#ifdef ROOT_MEAS_TIMING
  std::cout << "ROOT_MEAS_TIMING defined \n";
#endif
#ifdef ROOT_MATH_SYCL
  std::cout << "ROOT_MATH_SYCL defined \n";
#endif
#ifdef ROOT_MATH_CUDA
  std::cout << "ROOT_MATH_CUDA defined \n";
#endif
#ifdef SINGLE_PRECISION
  std::cout << "SINGLE_PRECISION defined \n";
#endif

  ROOT::EnableImplicitMT();

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  std::string arg2 = argv[2];
  std::size_t nruns = std::stoi(arg2, &pos);
  size_t local_size = 128;

  auto u_vectors = GenVectors(N);
  auto v_vectors = GenVectors(N);

  Vector<arithmetic_type> eta1(N, 1.), eta2(N, 1.), phi1(N, 1.), phi2(N, 1.),
      pt1(N, 1.), pt2(N, 1.), mass1(N, 1.), mass2(N, 1.);

  Vector<arithmetic_type> masses =
      InvariantMasses(u_vectors, v_vectors, N, local_size);

  Vector<arithmetic_type> masses2(N);

  for (size_t i = 0; i < nruns; i++)
  {
    auto start = std::chrono::system_clock::now();
    masses2 = ROOT::VecOps::InvariantMasses(
        pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2);

    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() *
        1e-6;
    std::cout << "RVec cpu time " << duration << " (s)" << std::endl;
  }
  assert((std::abs(masses[0] - 2.) <= 1e-5));
  assert((std::abs(masses2[0] - 2.) <= 1e-5));

  return 0;
}
////////////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include <climits>
#include <stdlib.h>

#include "SYCLMath/Cartesian2D.h"
#include "SYCLMath/Polar2D.h"
#include "SYCLMath/Vector2D.h"

#ifdef ROOT_MATH_SYCL
#include <sycl/sycl.hpp>
#endif

/*
// Helper function for toggling ON SYCL histogramming.
char env[] = "SYCL_HIST";
void EnableSYCL() { setenv(env, "1", 1); }

// Helper functions for element-wise comparison of histogram arrays.
#define CHECK_ARRAY(a, b, n)                                                   \
  {                                                                            \
    for (auto i : ROOT::TSeqI(n)) {                                            \
      EXPECT_EQ(a[i], b[i]) << "  at index i = " << i;                         \
    }                                                                          \
  }

#define CHECK_ARRAY_FLOAT(a, b, n)                                             \
  {                                                                            \
    for (auto i : ROOT::TSeqI(n)) {                                            \
      EXPECT_FLOAT_EQ(a[i], b[i]) << "  at index i = " << i;                   \
    }                                                                          \
  }

#define CHECK_ARRAY_DOUBLE(a, b, n)                                            \
  {                                                                            \
    for (auto i : ROOT::TSeqI(n)) {                                            \
      EXPECT_DOUBLE_EQ(a[i], b[i]) << "  at index i = " << i;                  \
    }                                                                          \
  }

template <typename T> void CompareArrays(T *result, T *expected, int n) {
  CHECK_ARRAY(result, expected, n)
}

template <> void CompareArrays(float *result, float *expected, int n) {
  CHECK_ARRAY_FLOAT(result, expected, n)
}

template <> void CompareArrays(double *result, double *expected, int n) {
  CHECK_ARRAY_DOUBLE(result, expected, n)
}
*/

using vec2d = ROOT::Experimental::DisplacementVector2D<
    ROOT::Experimental::Cartesian2D<double>>;

template <class V> void print_vec2d(V v, const char *name) {
  std::cout << name << " : " << v.x() << ", " << v.y() << std::endl;
}

int main() {
  // Cartesian2D point2dA{1.0,2.0}, point2dB{-1.0,-2.0}, point2dC;
  vec2d u{1.0, 2.0}, v{-2.0, 2.0}, w{};
 
  print_vec2d(u, "u");
  print_vec2d(v, "v");
  print_vec2d(w, "w");

#ifdef ROOT_MATH_SYCL
  sycl::default_selector device_selector;

  sycl::queue  queue(device_selector);
  // validation checks
  {

    std::cout << "sycl::queue check - selected device:\n"
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

        
      sycl::buffer<vec2d, 1> u_sycl(&u, sycl::range<1>(1));
      sycl::buffer<vec2d, 1> v_sycl(&v, sycl::range<1>(1));
      sycl::buffer<vec2d, 1> w_sycl(&w, sycl::range<1>(1));
  
      queue.submit([&] (sycl::handler& cgh) {
         auto u_acc = u_sycl.get_access<sycl::access::mode::read>(cgh);
         auto v_acc = v_sycl.get_access<sycl::access::mode::read>(cgh);
         auto w_acc = w_sycl.get_access<sycl::access::mode::discard_write>(cgh);

         cgh.single_task<class vector_addition>([=] () {
         w_acc[0] = u_acc[0] + v_acc[0];
         });
      });


  }
#else
w = u+v;
#endif
  print_vec2d(w,"w");

 }
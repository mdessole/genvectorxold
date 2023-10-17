////////////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include <climits>
#include <stdlib.h>

#include "Math/Cartesian2D.h"
#include "Math/Polar2D.h"
#include "Math/Vector2D.h"

using ROOT::Math;

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
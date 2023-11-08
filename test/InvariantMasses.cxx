
#include "SYCLMath/Vector4D.h"
#include "SYCLMath/VecOps.h"

#include "gtest/gtest.h"

// Helper functions for element-wise comparison of histogram arrays.
#define CHECK_ARRAY(a, b, n)                              \
   {                                                      \
      for (size_t=0; i<n; i++) {                     \
         EXPECT_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                   \
   }

#define CHECK_ARRAY_FLOAT(a, b, n)                              \
   {                                                            \
      for (size_t=0; i<n; i++) {                           \
         EXPECT_FLOAT_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                         \
   }

#define CHECK_ARRAY_DOUBLE(a, b, n)                              \
   {                                                             \
      for (size_t=0; i<n; i++) {                            \
         EXPECT_DOUBLE_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                          \
   }

template <typename T>
void CompareArrays(T *result, T *expected, int n)
{
   CHECK_ARRAY(result, expected, n)
}

template <>
void CompareArrays(float *result, float *expected, int n)
{
   CHECK_ARRAY_FLOAT(result, expected, n)
}

template <>
void CompareArrays(double *result, double *expected, int n)
{
   CHECK_ARRAY_DOUBLE(result, expected, n)
}


template <typename Scalar>
class InvMassesTestFixture : public ::testing::Test {
public:
using LVector = ROOT::Experimental::LorentzVector<
    ROOT::Experimental::PtEtaPhiM4D<Scalar>>;

};

using InvMassesTestTypes = ::testing::Types<double,float>;

TYPED_TEST_SUITE(InvMassesTestFixture, InvMassesTestTypes);

TYPED_TEST(InvMassesTestFixture, InvMasses)
{
   // int, double, or float
   using t = typename TestFixture::dataType;
   auto &h = this->histogram;

   std::vector<ROOT::RVecD> coords = {
      ROOT::RVecD(this->dim, this->startBin - 1),                   // Underflow
      ROOT::RVecD(this->dim, (this->startBin + this->endBin) / 2.), // Center
      ROOT::RVecD(this->dim, this->endBin + 1)                      // OVerflow
   };
   auto weight = (t)1;

   std::vector<int> expectedHistBins = {0, this->nCells / 2, this->nCells - 1};

   for (auto i = 0; i < (int)coords.size(); i++) {
      h.Fill(coords[i]);
      this->expectedHist[expectedHistBins[i]] = weight;
   }

   h.RetrieveResults(this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(coords, weight);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}
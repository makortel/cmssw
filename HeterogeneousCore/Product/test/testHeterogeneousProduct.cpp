#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

class testHeterogeneousProduct: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testHeterogeneousProduct);
  CPPUNIT_TEST(testDefault);
  CPPUNIT_TEST(testCPU);
  CPPUNIT_TEST(testGPU);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void testDefault();
  void testCPU();
  void testGPU();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testHeterogeneousProduct);

void testHeterogeneousProduct::testDefault() {
  HeterogeneousProduct<heterogeneous::CPUProduct<int>,
                       heterogeneous::GPUMockProduct<int, heterogeneous::CPUProduct<int>>
                       > prod;

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  prod.getProduct<heterogeneous::GPUMockProduct<int, heterogeneous::CPUProduct<int>>>();
  
  /*
  HeterogeneousProduct<int, int> prod;
  const auto& tmp = prod;
  
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kGPUCuda) == false);
  CPPUNIT_ASSERT_THROW(tmp.getCPUProduct(), cms::Exception);
  CPPUNIT_ASSERT_THROW(tmp.getGPUProduct(), cms::Exception);
  */
}

void testHeterogeneousProduct::testCPU() {
  /*
  HeterogeneousProduct<int, int> prod{5};
  const auto& tmp = prod;

  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kGPUCuda) == false);
  CPPUNIT_ASSERT(tmp.getCPUProduct() == 5);
  CPPUNIT_ASSERT_THROW(tmp.getGPUProduct(), cms::Exception);
  */
}

void testHeterogeneousProduct::testGPU() {
  /*
  HeterogeneousProduct<int, int> prod{5, [](const int& src, int& dst) { dst = src; }};
  const auto& tmp = prod;

  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kGPUCuda) == true);
  CPPUNIT_ASSERT(tmp.getGPUProduct() == 5);

  // Automatic transfer
  CPPUNIT_ASSERT(tmp.getCPUProduct() == 5);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(tmp.isProductOn(HeterogeneousDevice::kGPUCuda) == true);
  CPPUNIT_ASSERT(tmp.getGPUProduct() == 5);
  */
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

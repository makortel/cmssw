#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

class testHeterogeneousProduct: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testHeterogeneousProduct);
  CPPUNIT_TEST(testDefault);
  CPPUNIT_TEST(testCPU);
  CPPUNIT_TEST(testGPUMock);
  CPPUNIT_TEST(testGPUCuda);
  CPPUNIT_TEST(testGPUAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void testDefault();
  void testCPU();
  void testGPUMock();
  void testGPUCuda();
  void testGPUAll();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testHeterogeneousProduct);

void testHeterogeneousProduct::testDefault() {
  HeterogeneousProduct<heterogeneous::CPUProduct<int>,
                       heterogeneous::GPUMockProduct<int>
                       > prod;

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT_THROW(prod.getProduct<HeterogeneousDevice::kCPU>(), cms::Exception);
  CPPUNIT_ASSERT_THROW(prod.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
}

void testHeterogeneousProduct::testCPU() {
  HeterogeneousProduct<heterogeneous::CPUProduct<int>,
                       heterogeneous::GPUMockProduct<int>
                       > prod{heterogeneous::cpuProduct(5)};

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT_THROW(prod.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
}

void testHeterogeneousProduct::testGPUMock() {
  HeterogeneousProduct<heterogeneous::CPUProduct<int>,
                       heterogeneous::GPUMockProduct<int>
                       > prod{heterogeneous::gpuMockProduct(5),
                              [](const int& src, int& dst) { dst = src; }};

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kGPUMock>() == 5);

  // Automatic transfer
  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == false);
  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
}

void testHeterogeneousProduct::testGPUCuda() {
  HeterogeneousProduct<heterogeneous::CPUProduct<int>,
                       heterogeneous::GPUCudaProduct<int>
                       > prod{heterogeneous::gpuCudaProduct(5),
                              [](const int& src, int& dst) { dst = src; }};

  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == true);

  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);

  // Automatic transfer
  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod.isProductOn(HeterogeneousDevice::kGPUCuda) == true);
  CPPUNIT_ASSERT(prod.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);
}

void testHeterogeneousProduct::testGPUAll() {
  // Data initially on CPU
  HeterogeneousProduct<heterogeneous::CPUProduct<int>,
                       heterogeneous::GPUMockProduct<int>,
                       heterogeneous::GPUCudaProduct<int>
                       > prod1{heterogeneous::cpuProduct(5)};

  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod1.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod1.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT_THROW(prod1.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT_THROW(prod1.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // Data initially on GPUMock
  HeterogeneousProduct<heterogeneous::CPUProduct<int>,
                       heterogeneous::GPUMockProduct<int>,
                       heterogeneous::GPUCudaProduct<int>
                       > prod2{heterogeneous::gpuMockProduct(5),
                              [](const int& src, int& dst) { dst = src; }};

  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == false);

  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // Automatic transfer
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUMock) == true);
  CPPUNIT_ASSERT(prod2.isProductOn(HeterogeneousDevice::kGPUCuda) == false);
  CPPUNIT_ASSERT(prod2.getProduct<HeterogeneousDevice::kGPUMock>() == 5);
  CPPUNIT_ASSERT_THROW(prod2.getProduct<HeterogeneousDevice::kGPUCuda>(), cms::Exception);

  // Data initially on GPUCuda
  HeterogeneousProduct<heterogeneous::CPUProduct<int>,
                       heterogeneous::GPUMockProduct<int>,
                       heterogeneous::GPUCudaProduct<int>
                       > prod3{heterogeneous::gpuCudaProduct(5),
                              [](const int& src, int& dst) { dst = src; }};

  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kCPU) == false);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kGPUCuda) == true);

  CPPUNIT_ASSERT_THROW(prod3.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT(prod3.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);

  // Automatic transfer
  CPPUNIT_ASSERT(prod3.getProduct<HeterogeneousDevice::kCPU>() == 5);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kCPU) == true);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kGPUMock) == false);
  CPPUNIT_ASSERT(prod3.isProductOn(HeterogeneousDevice::kGPUCuda) == true);
  CPPUNIT_ASSERT_THROW(prod3.getProduct<HeterogeneousDevice::kGPUMock>(), cms::Exception);
  CPPUNIT_ASSERT(prod3.getProduct<HeterogeneousDevice::kGPUCuda>() == 5);
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp" // to be removed after rebase to 80X
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <iomanip>

#include "DataFormats/PatCandidates/interface/liblogintpack.h"

class testlogintpack : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testlogintpack);

  CPPUNIT_TEST(test);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void test();

private:
};

void testlogintpack::test() {
  auto pack = [](double x){ return logintpack::pack8log(x, -15, 0); };
  auto packceil = [](double x){ return logintpack::pack8log(x, -15, 0); };
  auto packclosed = [](double x){ return logintpack::pack8log(x, -15, 0); };
  auto unpack = [](int8_t x){ return logintpack::unpack8log(x, -15, 0); };
  auto unpackclosed = [](int8_t x){ return logintpack::unpack8logClosed(x, -15, 0); };

  CPPUNIT_ASSERT(pack(std::exp(-15.f)) == logintpack::smallestPositive);
  CPPUNIT_ASSERT(packceil(std::exp(-15.f)) == logintpack::smallestPositive);
  CPPUNIT_ASSERT(packclosed(std::exp(-15.f)) == logintpack::smallestPositive);
  CPPUNIT_ASSERT(unpack(logintpack::smallestPositive) == std::exp(-15.f));
  CPPUNIT_ASSERT(unpackclosed(logintpack::smallestPositive) == std::exp(-15.f));

  CPPUNIT_ASSERT(pack(-std::exp(-15.f)) == logintpack::smallestNegative);
  CPPUNIT_ASSERT(packceil(-std::exp(-15.f)) == logintpack::smallestNegative);
  CPPUNIT_ASSERT(unpack(logintpack::smallestNegative) == -std::exp(-15.f));
  CPPUNIT_ASSERT(unpack(pack(-std::exp(-15.f))) == -std::exp(-15.f));
  CPPUNIT_ASSERT(unpack(packceil(-std::exp(-15.f))) == -std::exp(-15.f));
  CPPUNIT_ASSERT(unpackclosed(packclosed(-std::exp(-15.f))) == -std::exp(-15.f));

  const float largestValue = std::exp(-15.f+127.f/128.f*15.f);
  CPPUNIT_ASSERT(pack(std::exp(0.f)) == 127); // this one actually overflows
  CPPUNIT_ASSERT(pack(largestValue) == 127);
  CPPUNIT_ASSERT(packceil(largestValue) == 127);
  CPPUNIT_ASSERT(unpack(127) == largestValue);

  CPPUNIT_ASSERT(pack(-largestValue) == -127);
  CPPUNIT_ASSERT(packceil(-largestValue) == -127);
  CPPUNIT_ASSERT(unpack(-127) == -largestValue);

  const float largestValueClosed = std::exp(0.f);
  CPPUNIT_ASSERT(packclosed(largestValueClosed) == 127);
  CPPUNIT_ASSERT(unpackclosed(127) == largestValueClosed);
  CPPUNIT_ASSERT(packclosed(-largestValueClosed) == -127);
  CPPUNIT_ASSERT(unpackclosed(-127) == -largestValueClosed);
}

CPPUNIT_TEST_SUITE_REGISTRATION(testlogintpack);


#include "Utilities/Testing/interface/CppUnit_testdriver.icpp" // to be removed after rebase to 80X
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <iomanip>
#include <limits>

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

namespace {
  double pack(double x)         { return logintpack::pack8log        (x, -15, 0); }
  double packceil(double x)     { return logintpack::pack8logCeil    (x, -15, 0); }
  double packclosed(double x)   { return logintpack::pack8log        (x, -15, 0); }
  double unpack(int8_t x)       { return logintpack::unpack8log      (x, -15, 0); }
  double unpackclosed(int8_t x) { return logintpack::unpack8logClosed(x, -15, 0); }
}

void testlogintpack::test() {
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

  const float someValue = std::exp(-15.f + 1/128.f*15.f);
  CPPUNIT_ASSERT(unpack(packceil(someValue)) == someValue);
  union { float flt; uint32_t i32; } conv;
  conv.flt = someValue;
  conv.i32 += 1;
  // The someValue+1..5 ulps fail currently, because of rounding in
  // float-log in pack8logCeil; to be enabled once the pack8logCeil gets fixed
  /*
  const float someValuePlus1Ulp = conv.flt;
  CPPUNIT_ASSERT(unpack(packceil(someValuePlus1Ulp)) >= someValuePlus1Ulp);
  */
  conv.i32 += 5;
  const float someValuePlus6Ulp = conv.flt;
  CPPUNIT_ASSERT(unpack(packceil(someValuePlus6Ulp)) >= someValuePlus6Ulp);

}

CPPUNIT_TEST_SUITE_REGISTRATION(testlogintpack);


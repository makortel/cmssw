#include <iostream>

#include "DataFormats/PatCandidates/interface/libminifloat.h"
#include "FWCore/Utilities/interface/isFinite.h"

bool testMax() {
  // 0x1f exponent is for inf, so 0x1e is the maximum
  // in maximum mantissa all bits are 1
  const uint16_t minifloatmax = (0x1e << 10) | 0x3ff;
  if(MiniFloatConverter::float16to32(minifloatmax) != MiniFloatConverter::max()) {
    std::cout << "MiniFloatConverter::max() does not correspond to mantissatable[offsettable[0x1e]+0x3ff]+exponenttable[0x1e] (" << MiniFloatConverter::float16to32(minifloatmax) << "), but " << MiniFloatConverter::max() << std::endl;
    return false;
  }

  // adding 1 ulp to max should give inf
  const uint16_t minifloatinf = minifloatmax + 1;
  if(edm::isFinite(MiniFloatConverter::float16to32(minifloatinf))) {
    std::cout << "MiniFloatConverter::max() + 1ulp does not yield inf but " << MiniFloatConverter::float16to32(minifloatinf) << std::endl;
    return false;
  }

  return true;
}

bool testMax32ConvertibleToMax16() {
  // max32ConvertibleToMax16() -> float16 -> float32 should be the same as max()
  const float max32ConvertedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(MiniFloatConverter::max32ConvertibleToMax16()));
  if(max32ConvertedTo16 != MiniFloatConverter::max()) {
    std::cout << "MiniFloatConverter::max32ConvertibleToMax16() converted to float16->float32 does not give MiniFloatConverter::max() (" << MiniFloatConverter::max() << "), but " << max32ConvertedTo16 << std::endl;
    return false;
  }

  // max32ConvertibleToMax16() + 1ulp should give inf
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::max32ConvertibleToMax16();
  conv.i32 += 1;
  const float max32PlusConvertedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(conv.flt));
  if(edm::isFinite(max32PlusConvertedTo16)) {
    std::cout << "MiniFloatConverter::max32ConvertibleToMax16() + 1ulp ->float16->float32 does not yield inf but " << max32PlusConvertedTo16 << std::endl;  }

  return true;
}

int main(void) {
  bool success = true;

  success = success && testMax();
  success = success && testMax32ConvertibleToMax16();

  return success ? 0 : 1;
}

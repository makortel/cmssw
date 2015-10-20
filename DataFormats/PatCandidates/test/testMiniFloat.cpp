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

bool testMax32RoundedToMax16() {
  // max32RoundedToMax16() -> float16 -> float32 should be the same as max()
  const float max32ConvertedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(MiniFloatConverter::max32RoundedToMax16()));
  if(max32ConvertedTo16 != MiniFloatConverter::max()) {
    std::cout << "MiniFloatConverter::max32RoundedToMax16() converted to float16->float32 does not give MiniFloatConverter::max() (" << MiniFloatConverter::max() << "), but " << max32ConvertedTo16 << std::endl;
    return false;
  }

  // max32RoundedToMax16() + 1ulp should give inf
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::max32RoundedToMax16();
  conv.i32 += 1;
  const float max32PlusConvertedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(conv.flt));
  if(edm::isFinite(max32PlusConvertedTo16)) {
    std::cout << "MiniFloatConverter::max32RoundedToMax16() + 1ulp ->float16->float32 does not yield inf but " << max32PlusConvertedTo16 << std::endl;
    return false;
  }

  return true;
}

bool testMin() {
  // 1 exponent, and 0 mantissa gives the smallest non-denormalized number of float16
  const uint16_t minifloat_min = 1 << 10;
  if(MiniFloatConverter::float16to32(minifloat_min) != MiniFloatConverter::min()) {
    std::cout << "MiniFloatConverter::min() does not correspond to mantissatable[offsettable[1]+0]+exponenttable[1] (" << MiniFloatConverter::float16to32(minifloat_min) << "), but " << MiniFloatConverter::min() << std::endl;
    return false;
  }

  // subtracting 1 ulp from min should give denormalized number, i.e. 0 exponent
  const uint16_t minifloat_denorm = MiniFloatConverter::float32to16(MiniFloatConverter::min()) - 1;
  if((minifloat_denorm >> 10) != 0) {
    std::cout << "MiniFloatConverter::min() - 1ulp does not yield denormalized number but " << MiniFloatConverter::float16to32(minifloat_denorm) << std::endl;
    return false;
  }

  // subtracking 1 ulp from float32 version of min should also give denormalized
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::min();
  conv.i32 -= 1;
  const uint16_t min32MinusConvertedTo16 = MiniFloatConverter::float32to16crop(conv.flt);
  if((min32MinusConvertedTo16 >> 10) != 0) {
    std::cout << "MiniFloatConverter::min() - 1ulp ->float16crop does not yield denormalized number but 0x" << std::hex << min32MinusConvertedTo16 << std::endl;  
    return false;
  }

  return true;
}

bool testMin32RoundedToMin16() {
  // min32RoundedToMin16() -> float16 -> float32 should be the same as min()
  const float min32RoundedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(MiniFloatConverter::min32RoundedToMin16()));
  if(min32RoundedTo16 != MiniFloatConverter::min()) {
    std::cout << "MiniFloatConverter::min32RoundedToMin16() converted to float16->float32 does not give MiniFloatConverter::min() (" << MiniFloatConverter::min() << "), but "
              << min32RoundedTo16 << " 0x" << std::hex << MiniFloatConverter::float32to16(MiniFloatConverter::min32RoundedToMin16()) << std::endl;
    return false;
  }

  // min32RoundedToMax16() - 1ulp should give denormalized
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::min32RoundedToMin16();
  conv.i32 -= 1;
  const uint16_t min32MinusRoundedTo16 = MiniFloatConverter::float32to16(conv.flt);
  if((min32MinusRoundedTo16 >> 10) != 0) {
    std::cout << "MiniFloatConverter::min32RoundedToMin16() - 1ulp ->float16 does not yield denormalized number but "
              << min32MinusRoundedTo16 << " 0x" << std::hex << min32MinusRoundedTo16 << std::endl;
    return false;
  }

  return true;
}

bool testDenormMin() {
  // zero exponent, and 0x1 in mantissa gives the smallest number of
  // float16
  const uint16_t minifloat_denorm_min = 1;
  if(MiniFloatConverter::float16to32(minifloat_denorm_min) != MiniFloatConverter::denorm_min()) {
    std::cout << "MiniFloatConverter::denorm_min() does not correspond to mantissatable[offsettable[0]+1]+exponenttable[0x1e] (" << MiniFloatConverter::float16to32(minifloat_denorm_min) << "), but " << MiniFloatConverter::denorm_min() << std::endl;
    return false;
  }

  // subtracting 1 ulp from denorm_min should give 0
  const uint16_t minifloat0 = MiniFloatConverter::float32to16(MiniFloatConverter::denorm_min()) - 1;
  if(MiniFloatConverter::float16to32(minifloat0) != 0.f) {
    std::cout << "MiniFloatConverter::denorm_min() - 1ulp does not yield 0 but " << MiniFloatConverter::float16to32(minifloat0) <<  std::endl;
    return false;
  }

  // subtracking 1 ulp from float32 version of denorm_min should also give 0
  union { float flt; uint32_t i32; } conv;
  conv.flt = MiniFloatConverter::denorm_min();
  conv.i32 -= 1;
  const float min32MinusConvertedTo16 = MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(conv.flt));
  if(min32MinusConvertedTo16 != 0.f) {
    std::cout << "MiniFloatConverter::denorm_min() - 1ulp ->float16->float32 does not yield 0 but " << min32MinusConvertedTo16 << std::endl;  
    return false;
  }

  return true;
}

int main(void) {
  bool success = true;

  success = success && testMax();
  success = success && testMax32RoundedToMax16();

  success = success && testMin();
  success = success && testMin32RoundedToMin16();
  success = success && testDenormMin();

  return success ? 0 : 1;
}

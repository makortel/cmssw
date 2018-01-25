#include "DataFormats/Common/interface/Wrapper.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

namespace {
  struct dictionary {
    HeterogeneousProduct<unsigned int, unsigned int> hpuu;
    HeterogeneousProduct<unsigned int, int*> hpuup;
  };
}

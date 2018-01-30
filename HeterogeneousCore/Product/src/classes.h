#include "DataFormats/Common/interface/Wrapper.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include <utility>

namespace {
  struct dictionary {
    HeterogeneousProduct<unsigned int, unsigned int> hpuu;
    HeterogeneousProduct<unsigned int, int*> hpuip;
    HeterogeneousProduct<unsigned int, float*> hpufp;
    HeterogeneousProduct<unsigned int, std::pair<float*, float*>> hpupfpfp;
  };
}

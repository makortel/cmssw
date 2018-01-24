#include "DataFormats/Common/interface/Wrapper.h"
#include "TestGPU/AcceleratorService/interface/HeterogeneousProduct.h"

namespace {
    struct dictionary {
      HeterogeneousProduct<unsigned int, unsigned int> hpuu;
    };
}

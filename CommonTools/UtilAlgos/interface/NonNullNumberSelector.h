#ifndef UtilAlgos_NonNullNumberSelector_h
#define UtilAlgos_NonNullNumberSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/NonNullNumberSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<NonNullNumberSelector> {
      static NonNullNumberSelector make( const edm::ParameterSet & cfg ) {
	return NonNullNumberSelector();
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) {}
    };

  }
}

#endif


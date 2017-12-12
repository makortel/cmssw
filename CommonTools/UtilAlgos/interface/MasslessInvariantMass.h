#ifndef UtilAlgos_MasslessInvariantMass_h
#define UtilAlgos_MasslessInvariantMass_h
#include "CommonTools/Utils/interface/MasslessInvariantMass.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<MasslessInvariantMass> {
      static MasslessInvariantMass make( const edm::ParameterSet & cfg ) {
	return MasslessInvariantMass();
      }
      static void fillPSetDescription(edm::ParameterSetDescription& desc) {}
    };
    
  }
}



#endif


#ifndef UtilAlgos_RefSelector_h
#define UtilAlgos_RefSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/RefSelector.h"

namespace reco {
  namespace modules {

    template<typename S>
    struct ParameterAdapter<RefSelector<S> > {
      static RefSelector<S> make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return RefSelector<S>( modules::make<S>( cfg, iC ) );
      }
      static void fillPSetDescription(edm::ParameterSetDescription& desc) {
        modules::fillPSetDescription<S>(desc);
      }
    };

  }
}

#endif


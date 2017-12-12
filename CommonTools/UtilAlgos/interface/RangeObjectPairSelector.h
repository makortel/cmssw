#ifndef UtilAlgos_RangeObjectPairSelector_h
#define UtilAlgos_RangeObjectPairSelector_h
#include "CommonTools/Utils/interface/RangeObjectPairSelector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace reco {
  namespace modules {
    
    template<typename F>
    struct ParameterAdapter<RangeObjectPairSelector<F> > {
      static RangeObjectPairSelector<F> make( const edm::ParameterSet & cfg ) {
	return RangeObjectPairSelector<F>( cfg.template getParameter<double>( "rangeMin" ),
					   cfg.template getParameter<double>( "rangeMax" ),
					   reco::modules::make<F>( cfg )
					   );
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) {
        desc.add<double>("rangeMin", 0.);
        desc.add<double>("rangeMax", 0);
        reco::modules::fillPSetDescription<F>(desc);
      }
    };
    
  }
}



#endif


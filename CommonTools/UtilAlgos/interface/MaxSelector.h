#ifndef UtilAlgos_MaxSelector_h
#define UtilAlgos_MaxSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MaxSelector.h"

namespace reco {
  namespace modules {

    template<typename T>
    struct ParameterAdapter<MaxSelector<T> > {
      static MaxSelector<T> make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return MaxSelector<T>( cfg.template getParameter<double>( "max" ) );
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) {
        desc.add<double>("max", 0.);
      }
    };

  }
}

#endif


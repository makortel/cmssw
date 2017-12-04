#ifndef UtilAlgos_NullPostProcessor_h
#define UtilAlgos_NullPostProcessor_h
/* \class helper::NullPostProcessor<OutputCollection>
 *
 * \author Luca Lista, INFN
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

namespace edm {
  class EDFilter;
  class Event;
}

namespace helper {

  template<typename OutputCollection, typename EdmFilter=edm::EDFilter>
  struct NullPostProcessor {
    NullPostProcessor( const edm::ParameterSet & iConfig, edm::ConsumesCollector && iC ) :
      NullPostProcessor( iConfig ) { }
    NullPostProcessor( const edm::ParameterSet & iConfig ) { }
    static void fillPSetDescription(edm::ParameterSetDescription& desc) {}
    void init( EdmFilter & ) { }
    void process( edm::OrphanHandle<OutputCollection>, edm::Event & ) { }
  };

}

#endif


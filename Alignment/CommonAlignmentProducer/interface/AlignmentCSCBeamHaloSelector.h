#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentCSCBeamHaloSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentCSCBeamHaloSelector_h

#include <vector>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace edm {
   class Event;
}

class TrackingRecHit;

class AlignmentCSCBeamHaloSelector {
   public:
      typedef std::vector<const reco::Track*> Tracks;

      /// constructor
      AlignmentCSCBeamHaloSelector(const edm::ParameterSet &iConfig, edm::ConsumesCollector & iC);

      /// destructor
      ~AlignmentCSCBeamHaloSelector();

      static void fillPSetDescription(edm::ParameterSetDescription& desc);

      /// select tracks
      Tracks select(const Tracks &tracks, const edm::Event &iEvent) const;

   private:
      unsigned int m_minStations;
      unsigned int m_minHitsPerStation;
};

#endif

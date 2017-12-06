#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentCSCOverlapSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentCSCOverlapSelector_h

#include <vector>
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace edm {
   class Event;
}

class TrackingRecHit;

class AlignmentCSCOverlapSelector {
   public:
      typedef std::vector<const reco::Track*> Tracks; 

      /// constructor
      AlignmentCSCOverlapSelector(const edm::ParameterSet &iConfig);

      /// destructor
      ~AlignmentCSCOverlapSelector();

      static void fillPSetDescription(edm::ParameterSetDescription& desc);

      /// select tracks
      Tracks select(const Tracks &tracks, const edm::Event &iEvent) const;

   private:
      int m_station;
      unsigned int m_minHitsPerChamber;
};

#endif

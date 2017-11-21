
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

#include <cmath>

using namespace reco ;

GsfElectronCore::GsfElectronCore
 ()
 : ctfGsfOverlap_(0.), isEcalDrivenSeed_(false), isTrackerDrivenSeed_(false)
 {}

 GsfElectronCore::GsfElectronCore
  ( const GsfTrackRef & gsfTrack )
  : gsfTrack_(gsfTrack), ctfGsfOverlap_(0.), isEcalDrivenSeed_(false), isTrackerDrivenSeed_(false)
  {
   const TrajectorySeed *seed = gsfTrack_->extra()->seedPtr() ;
   if (seed == nullptr)
    { edm::LogError("GsfElectronCore")<<"The GsfTrack has no seed ?!" ; }
   else
    {
     const ElectronSeed *elseed = dynamic_cast<const ElectronSeed *>(seed);
     if (elseed == nullptr)
      { edm::LogError("GsfElectronCore")<<"The GsfTrack seed is not an ElectronSeed ?!" ; }
     else
      {
       if (elseed->isEcalDriven()) isEcalDrivenSeed_ = true ;
       if (elseed->isTrackerDriven()) isTrackerDrivenSeed_ = true ;
      }
    }
  }

GsfElectronCore * GsfElectronCore::clone() const
 { return new GsfElectronCore(*this) ; }

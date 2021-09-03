#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class MultipleScatteringParametrisationMakerESProducer : public edm::ESProducer {
public:
  MultipleScatteringParametrisationMakerESProducer(edm::ParameterSet const& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // Using NavigationSchoolRecord here sounds a bit confusing, that
  // that depends only on TrackerRecoGeometryRecord and
  // IdealMagneticFieldRecord
  std::unique_ptr<MultipleScatteringParametrisationMaker> produce(const NavigationSchoolRecord& iRecord);

private:
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> trackerToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfieldToken_;
};

MultipleScatteringParametrisationMakerESProducer::MultipleScatteringParametrisationMakerESProducer(
    edm::ParameterSet const& iConfig) {
  auto cc = setWhatProduced(this);
  trackerToken_ = cc.consumes();
  bfieldToken_ = cc.consumes();
}

void MultipleScatteringParametrisationMakerESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<MultipleScatteringParametrisationMaker> MultipleScatteringParametrisationMakerESProducer::produce(
    const NavigationSchoolRecord& iRecord) {
  return std::make_unique<MultipleScatteringParametrisationMaker>(iRecord.get(trackerToken_),
                                                                  iRecord.get(bfieldToken_));
}

DEFINE_FWK_EVENTSETUP_MODULE(MultipleScatteringParametrisationMakerESProducer);

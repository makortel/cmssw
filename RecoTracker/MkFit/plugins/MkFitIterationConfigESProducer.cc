#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitIterationConfig.h"

// mkFit includes
#include "mkFit/IterationConfig.h"

class MkFitIterationConfigESProducer : public edm::ESProducer {
public:
  MkFitIterationConfigESProducer(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<MkFitIterationConfig> produce(const TrackerRecoGeometryRecord& iRecord);

private:
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> geomToken_;
  const std::string configFile_;
};

MkFitIterationConfigESProducer::MkFitIterationConfigESProducer(const edm::ParameterSet& iConfig)
    : geomToken_{setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName")).consumes()},
      configFile_{iConfig.getParameter<edm::FileInPath>("config").fullPath()} {}

void MkFitIterationConfigESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName")->setComment("Product label");
  desc.add<edm::FileInPath>("config")->setComment("Path to the JSON file for the mkFit configuration parameters");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<MkFitIterationConfig> MkFitIterationConfigESProducer::produce(const TrackerRecoGeometryRecord& iRecord) {
  auto const& geom = iRecord.get(geomToken_);
  // copy to avoid writes to shared object
  auto itsInfo = std::make_unique<mkfit::IterationsInfo>(geom.iterationsInfo());
  auto const* itConfig = &mkfit::ConfigJson_Load_File(*itsInfo, configFile_);

  return std::make_unique<MkFitIterationConfig>(std::move(itsInfo), itConfig);
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitIterationConfigESProducer);

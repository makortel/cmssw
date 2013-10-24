#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelClusterShapeCache.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include <cassert>

class SiPixelClusterShapeCacheProducer: public edm::EDProducer {
public:
  explicit SiPixelClusterShapeCacheProducer(const edm::ParameterSet& iConfig);
  ~SiPixelClusterShapeCacheProducer();

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  edm::InputTag src_;
};

SiPixelClusterShapeCacheProducer::SiPixelClusterShapeCacheProducer(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src"))
{
  produces<SiPixelClusterShapeCache>();
}

SiPixelClusterShapeCacheProducer::~SiPixelClusterShapeCacheProducer() {}

void SiPixelClusterShapeCacheProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > input;
  iEvent.getByLabel(src_, input);

  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get(geom);

  std::auto_ptr<SiPixelClusterShapeCache> output(new SiPixelClusterShapeCache(input));
  output->resize(input->data().size());

  ClusterData data; // reused
  ClusterShape clusterShape;

  for(const auto& detSet: *input) {
    const GeomDetUnit *genericDet = geom->idToDetUnit(detSet.detId());
    const PixelGeomDetUnit *pixDet = dynamic_cast<const PixelGeomDetUnit *>(genericDet);
    assert(pixDet);

    edmNew::DetSet<SiPixelCluster>::const_iterator iCluster = detSet.begin(), endCluster = detSet.end();
    for(; iCluster != endCluster; ++iCluster) {
      clusterShape.determineShape(*pixDet, *iCluster, data);
      SiPixelClusterShapeCache::ClusterRef clusterRef = edmNew::makeRefTo(input, iCluster);
      output->set(clusterRef, data);
      data.size.clear();
    }
  }

  iEvent.put(output);
}

DEFINE_FWK_MODULE(SiPixelClusterShapeCacheProducer);

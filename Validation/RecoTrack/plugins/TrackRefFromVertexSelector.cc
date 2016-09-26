#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace reco {
  typedef edm::RefToBaseVector<reco::Track> TrackBaseRefVector;
}

class TrackRefFromVertexSelector: public edm::EDProducer {
public:
  explicit TrackRefFromVertexSelector(const edm::ParameterSet& iConfig);
  ~TrackRefFromVertexSelector();

private:
  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
      
  const edm::InputTag vertexSrc_;
  const double weightCut_;
};


TrackRefFromVertexSelector::TrackRefFromVertexSelector(const edm::ParameterSet& iConfig):
  vertexSrc_(iConfig.getParameter<edm::InputTag>("vertexSrc")),
  weightCut_(iConfig.getParameter<double>("weightCut"))
{
  produces<reco::TrackBaseRefVector>();
}

TrackRefFromVertexSelector::~TrackRefFromVertexSelector() {}

void TrackRefFromVertexSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::VertexCollection> vertexH;
  iEvent.getByLabel(vertexSrc_, vertexH);
  const reco::VertexCollection& vertices = *vertexH;

  std::auto_ptr<reco::TrackBaseRefVector> ret(new reco::TrackBaseRefVector());
  if(!vertices.empty()) {
    const auto& pv = vertices[0];

    for(auto iTrack = pv.tracks_begin(); iTrack != pv.tracks_end(); ++iTrack) {
      if(pv.trackWeight(*iTrack) > weightCut_) {
        ret->push_back(*iTrack);
      }
    }
  }

  iEvent.put(ret);  
}

DEFINE_FWK_MODULE(TrackRefFromVertexSelector);

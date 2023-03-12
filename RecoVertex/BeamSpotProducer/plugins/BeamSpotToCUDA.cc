#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

namespace {

  class BeamSpotHost {
  public:
    BeamSpotHost() : data_h_{cms::cuda::make_host_noncached_unique<BeamSpotPOD>(cudaHostAllocWriteCombined)} {}

    BeamSpotHost(BeamSpotHost const&) = delete;
    BeamSpotHost(BeamSpotHost&&) = default;

    BeamSpotHost& operator=(BeamSpotHost const&) = delete;
    BeamSpotHost& operator=(BeamSpotHost&&) = default;

    BeamSpotPOD* data() { return data_h_.get(); }
    BeamSpotPOD const* data() const { return data_h_.get(); }

    cms::cuda::host::noncached::unique_ptr<BeamSpotPOD>& ptr() { return data_h_; }
    cms::cuda::host::noncached::unique_ptr<BeamSpotPOD> const& ptr() const { return data_h_; }

  private:
    cms::cuda::host::noncached::unique_ptr<BeamSpotPOD> data_h_;
  };

}  // namespace

class BeamSpotToCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit BeamSpotToCUDA(const edm::ParameterSet& iConfig);
  ~BeamSpotToCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginStream(edm::StreamID) override {
    edm::Service<CUDAService> cs;
    if (cs->enabled()) {
      beamSpotHost_ = std::make_unique<BeamSpotHost>();
    }
  }
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  const edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  const edm::EDPutTokenT<cms::cuda::Product<BeamSpotCUDA>> bsPutToken_;
  std::unique_ptr<BeamSpotHost> beamSpotHost_;
  BeamSpotCUDA beamSpotDevice_;
  cms::cuda::ContextState ctxState_;
};

BeamSpotToCUDA::BeamSpotToCUDA(const edm::ParameterSet& iConfig)
    : bsGetToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("src"))},
      bsPutToken_{produces<cms::cuda::Product<BeamSpotCUDA>>()} {}

void BeamSpotToCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("offlineBeamSpot"));
  descriptions.add("offlineBeamSpotToCUDA", desc);
}

void BeamSpotToCUDA::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};

  const reco::BeamSpot& bs = iEvent.get(bsGetToken_);

  auto& bsHost = beamSpotHost_->ptr();

  bsHost->x = bs.x0();
  bsHost->y = bs.y0();
  bsHost->z = bs.z0();

  bsHost->sigmaZ = bs.sigmaZ();
  bsHost->beamWidthX = bs.BeamWidthX();
  bsHost->beamWidthY = bs.BeamWidthY();
  bsHost->dxdz = bs.dxdz();
  bsHost->dydz = bs.dydz();
  bsHost->emittanceX = bs.emittanceX();
  bsHost->emittanceY = bs.emittanceY();
  bsHost->betaStar = bs.betaStar();

  BeamSpotCUDA bsDevice(ctx.stream());
  cms::cuda::copyAsync(bsDevice.ptr(), bsHost, ctx.stream());
  beamSpotDevice_ = std::move(bsDevice);
}

void BeamSpotToCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};
  ctx.emplace(iEvent, bsPutToken_, std::move(beamSpotDevice_));
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);

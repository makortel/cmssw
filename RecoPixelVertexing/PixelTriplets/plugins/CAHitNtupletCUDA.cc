#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"

class CAHitNtupletCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit CAHitNtupletCUDA(const edm::ParameterSet& iConfig);
  ~CAHitNtupletCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void endStream() override;

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  bool onGPU_;

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DGPU>> tokenHitGPU_;
  edm::EDPutTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenTrackGPU_;
  edm::EDGetTokenT<TrackingRecHit2DCPU> tokenHitCPU_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenTrackCPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;

  PixelTrackHeterogeneous data_;
  cms::cuda::ContextState ctxState_;
};

CAHitNtupletCUDA::CAHitNtupletCUDA(const edm::ParameterSet& iConfig)
    : onGPU_(iConfig.getParameter<bool>("onGPU")), tokenField_(esConsumes()), gpuAlgo_(iConfig, consumesCollector()) {
  if (onGPU_) {
    tokenHitGPU_ =
        consumes<cms::cuda::Product<TrackingRecHit2DGPU>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenTrackGPU_ = produces<cms::cuda::Product<PixelTrackHeterogeneous>>();
  } else {
    tokenHitCPU_ = consumes<TrackingRecHit2DCPU>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenTrackCPU_ = produces<PixelTrackHeterogeneous>();
  }
}

void CAHitNtupletCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));

  CAHitNtupletGeneratorOnGPU::fillDescriptions(desc);
  descriptions.add("pixelTracksCUDA", desc);
}

void CAHitNtupletCUDA::beginStream(edm::StreamID) { gpuAlgo_.beginJob(); }

void CAHitNtupletCUDA::endStream() { gpuAlgo_.endJob(); }

void CAHitNtupletCUDA::acquire(const edm::Event& iEvent, const edm::EventSetup& es, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

  if (onGPU_) {
    auto hHits = iEvent.getHandle(tokenHitGPU_);

    cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};
    auto const& hits = ctx.get(*hHits);

    data_ = gpuAlgo_.makeTuplesAsync(hits, bf, ctx.stream());
  } else {
    auto const& hits = iEvent.get(tokenHitCPU_);
    data_ = gpuAlgo_.makeTuples(hits, bf);
    auto holder = waitingTaskHolder.makeWaitingTaskHolderAndRelease();
    holder.doneWaiting(nullptr);
  }
}

void CAHitNtupletCUDA::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  if (onGPU_) {
    cms::cuda::ScopedContextProduce ctx{ctxState_};
    ctx.emplace(iEvent, tokenTrackGPU_, std::move(data_));
  } else {
    iEvent.emplace(tokenTrackCPU_, std::move(data_));
  }
}

DEFINE_FWK_MODULE(CAHitNtupletCUDA);

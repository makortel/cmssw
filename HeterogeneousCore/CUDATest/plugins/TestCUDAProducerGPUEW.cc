#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include "TestCUDAProducerGPUKernel.h"

class TestCUDAProducerGPUEW : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerGPUEW(edm::ParameterSet const& iConfig);
  ~TestCUDAProducerGPUEW() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

private:
  std::string const label_;
  edm::EDGetTokenT<CUDAProduct<CUDAThing>> const srcToken_;
  edm::EDPutTokenT<CUDAProduct<CUDAThing>> const dstToken_;
  TestCUDAProducerGPUKernel gpuAlgo_;
  CUDAContextState ctxState_;
  cudautils::device::unique_ptr<float[]> devicePtr_;
  cudautils::host::noncached::unique_ptr<float> hostData_;
};

TestCUDAProducerGPUEW::TestCUDAProducerGPUEW(edm::ParameterSet const& iConfig)
    : label_{iConfig.getParameter<std::string>("@module_label")},
      srcToken_{consumes<CUDAProduct<CUDAThing>>(iConfig.getParameter<edm::InputTag>("src"))},
      dstToken_{produces<CUDAProduct<CUDAThing>>()} {
  edm::Service<CUDAService> cs;
  if (cs->enabled()) {
    hostData_ = cudautils::make_host_noncached_unique<float>();
  }
}

void TestCUDAProducerGPUEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment(
      "This EDProducer is part of the TestCUDAProducer* family. It models a GPU algorithm this is not the first "
      "algorithm in the chain of the GPU EDProducers, and that transfers some data from GPU to CPU and thus needs to "
      "synchronize GPU and CPU. The synchronization is implemented with the ExternalWork extension. Produces "
      "CUDAProduct<CUDAThing>.");
}

void TestCUDAProducerGPUEW::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::LogVerbatim("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::acquire begin event "
                                            << iEvent.id().event() << " stream " << iEvent.streamID();

  auto const& in = iEvent.get(srcToken_);
  CUDAScopedContextAcquire ctx{in, std::move(waitingTaskHolder), ctxState_};
  CUDAThing const& input = ctx.get(in);

  devicePtr_ = gpuAlgo_.runAlgo(label_, input.get(), ctx.stream());
  // Mimick the need to transfer some of the GPU data back to CPU to
  // be used for something within this module, or to be put in the
  // event.
  cudaCheck(
      cudaMemcpyAsync(hostData_.get(), devicePtr_.get() + 10, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
  edm::LogVerbatim("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::acquire end event "
                                            << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestCUDAProducerGPUEW::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  edm::LogVerbatim("TestCUDAProducerGPUEW")
      << label_ << " TestCUDAProducerGPUEW::produce begin event " << iEvent.id().event() << " stream "
      << iEvent.streamID() << " 10th element " << *hostData_;

  CUDAScopedContextProduce ctx{ctxState_};

  ctx.emplace(iEvent, dstToken_, std::move(devicePtr_));

  edm::LogVerbatim("TestCUDAProducerGPUEW") << label_ << " TestCUDAProducerGPUEW::produce end event "
                                            << iEvent.id().event() << " stream " << iEvent.streamID();
}

DEFINE_FWK_MODULE(TestCUDAProducerGPUEW);

#include "CUDADataFormats/Common/interface/host_unique_ptr.h"
#include "CUDADataFormats/Common/interface/CUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"


class SiPixelDigisSoAFromCUDA: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelDigisSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelDigisSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<CUDA<SiPixelDigisCUDA>> digiGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  edm::cuda::host::unique_ptr<uint32_t[]> pdigi_;
  edm::cuda::host::unique_ptr<uint32_t[]> rawIdArr_;
  edm::cuda::host::unique_ptr<uint16_t[]> adc_;
  edm::cuda::host::unique_ptr< int32_t[]> clus_;

  edm::cuda::host::unique_ptr<PixelErrorCompact[]> data_;
  const GPU::SimpleVector<PixelErrorCompact> *error_ = nullptr;
  const PixelFormatterErrors *formatterErrors_ = nullptr;

  CUDAContextToken ctxTmp_;

  int nDigis_;
  bool includeErrors_;
};

SiPixelDigisSoAFromCUDA::SiPixelDigisSoAFromCUDA(const edm::ParameterSet& iConfig):
  digiGetToken_(consumes<CUDA<SiPixelDigisCUDA>>(iConfig.getParameter<edm::InputTag>("src"))),
  digiPutToken_(produces<SiPixelDigisSoA>())
{}

void SiPixelDigisSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigisSoAFromCUDA::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Do the transfer in a CUDA stream parallel to the computation CUDA stream
  auto ctx = CUDAScopedContext(iEvent.streamID(), std::move(waitingTaskHolder));

  edm::Handle<CUDA<SiPixelDigisCUDA>> hdigi;
  iEvent.getByToken(digiGetToken_, hdigi);
  const auto& gpuDigis = ctx.get(*hdigi);

  nDigis_ = gpuDigis.nDigis();
  pdigi_ = gpuDigis.pdigiToHostAsync(ctx.stream());
  rawIdArr_ = gpuDigis.rawIdArrToHostAsync(ctx.stream());
  adc_ = gpuDigis.adcToHostAsync(ctx.stream());
  clus_ = gpuDigis.clusToHostAsync(ctx.stream());

  includeErrors_ = gpuDigis.hasErrors();
  if(includeErrors_) {
    auto tmp = gpuDigis.dataErrorToHostAsync(ctx.stream());
    data_ = std::move(tmp.first);
    error_ = tmp.second;
    formatterErrors_ = &(gpuDigis.formatterErrors());
  }

  ctxTmp_ = ctx.toToken(); // CUDA stream must live until produce
}

void SiPixelDigisSoAFromCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // The following line copies the data from the pinned host memory to
  // regular host memory. In principle that feels unnecessary (why not
  // just use the pinned host memory?). There are a few arguments for
  // doing it though
  // - Now can release the pinned host memory back to the (caching) allocator
  //   * if we'd like to keep the pinned memory, we'd need to also
  //     keep the CUDA stream around as long as that, or allow pinned
  //     host memory to be allocated without a CUDA stream
  // - What if a CPU algorithm would produce the same SoA? We can't
  //   use cudaMallocHost without a GPU...
  if(includeErrors_) {
    iEvent.emplace(digiPutToken_, nDigis_, pdigi_.get(), rawIdArr_.get(), adc_.get(), clus_.get(),
                   error_->size(), error_->data(), formatterErrors_);
  }
  else {
    iEvent.emplace(digiPutToken_, nDigis_, pdigi_.get(), rawIdArr_.get(), adc_.get(), clus_.get());
  }

  pdigi_.reset();
  rawIdArr_.reset();
  adc_.reset();
  clus_.reset();
  data_.reset();
  error_ = nullptr;
  formatterErrors_ = nullptr;
  
  ctxTmp_.reset(); // release CUDA stream etc
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromCUDA);

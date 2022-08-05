#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_stream_SynchronizingEDProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_stream_SynchronizingEDProducer_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceEvent.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceEventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataAcquireSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace stream {
    template <typename... Args>
    class SynchronizingEDProducer : public ProducerBase<edm::stream::EDProducer, edm::ExternalWork, Args...> {
      static_assert(
          not edm::CheckAbility<edm::module::Abilities::kExternalWork, Args...>::kHasIt,
          "ExternalWork ability is redundant with ALPAKA_ACCELERATOR_NAMESPACE::stream::SynchronizingEDProducer."
          "Please remove it.");

    public:
      void acquire(edm::Event const& iEvent,
                   edm::EventSetup const& iSetup,
                   edm::WaitingTaskWithArenaHolder holder) final {
        detail::EDMetadataAcquireSentry sentry(iEvent.streamID(), std::move(holder));
        DeviceEvent const ev(iEvent, sentry.metadata());
        DeviceEventSetup const es(iSetup);
        acquire(ev, es);
        metadata_ = sentry.finish();
      }

      void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
        detail::EDMetadataSentry sentry(std::move(metadata_));
        DeviceEvent ev(iEvent, sentry.metadata());
        DeviceEventSetup const es(iSetup);
        produce(ev, es);
        sentry.finish();
      }

      virtual void acquire(DeviceEvent const& iEvent, DeviceEventSetup const& iSetup) = 0;

      virtual void produce(DeviceEvent& iEvent, DeviceEventSetup const& iSetup) = 0;

    private:
      std::shared_ptr<EDMetadata> metadata_;
    };
  }  // namespace stream
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

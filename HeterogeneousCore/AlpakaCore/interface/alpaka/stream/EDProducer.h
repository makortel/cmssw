#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_stream_EDProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_stream_EDProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceEvent.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceEventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace stream {
    template <typename... Args>
    class EDProducer : public ProducerBase<edm::stream::EDProducer, Args...> {
      static_assert(not edm::CheckAbility<edm::module::Abilities::kExternalWork, Args...>::kHasIt,
                    "ALPAKA_ACCELERATOR_NAMESPACE::stream::EDProducer may not be used with ExternalWork ability. "
                    "Please use ALPAKA_ACCELERATOR_NAMESPACE::stream::SynchronizingEDProducer instead.");

    public:
      void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
        detail::EDMetadataSentry sentry(iEvent.streamID());
        DeviceEvent ev(iEvent, sentry.metadata());
        DeviceEventSetup const es(iSetup);
        produce(ev, es);
        sentry.finish();
      }

      virtual void produce(DeviceEvent& iEvent, DeviceEventSetup const& iSetup) = 0;
    };
  }  // namespace stream
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/PluginFactoryService.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <memory>

namespace edmtest {
  class DummyBase {
  public:
    explicit DummyBase(const edm::ParameterSet& iConfig) {}
    virtual ~DummyBase() {}

    virtual int get() const = 0;
  };

  class Dummy1: public DummyBase {
  public:
    Dummy1(const edm::ParameterSet& iConfig, int ret): DummyBase(iConfig), ret_(ret) {}
    ~Dummy1() {}

    int get() const override {
      return ret_;
    }

  private:
    const int ret_;
  };

  class Dummy2: public DummyBase {
  public:
    Dummy2(const edm::ParameterSet& iConfig, int ret): DummyBase(iConfig), ret_(ret) {
      if(iConfig.getParameter<bool>("fromConfig")) {
        ret_ = iConfig.getParameter<int>("value");
      }
    }
    ~Dummy2() {}

    int get() const override {
      return ret_;
    }
  private:
    int ret_;                                                       
  };

  template <typename T, typename L>
  void test_equal(T test, T reference, const std::string& label, L lineno) {
    if(test != reference) {
      std::cerr << "TestPluginFactoryService (" << label << ", L " << lineno << "): " << "Expected value " << reference << ", got " << test << std::endl;
      abort();
    }
  }
}

typedef edmplugin::PluginFactory<edmtest::DummyBase *(const edm::ParameterSet&, int)> DummyFactory;
EDM_REGISTER_PLUGINFACTORY(DummyFactory, "DummyFactory");
DEFINE_EDM_PLUGIN(DummyFactory, edmtest::Dummy1, "Dummy1");
DEFINE_EDM_PLUGIN(DummyFactory, edmtest::Dummy2, "Dummy2");

namespace edmtest {
  class TestPluginFactoryService: public edm::EDAnalyzer {
  public:
    explicit TestPluginFactoryService(const edm::ParameterSet& iConfig);
    virtual ~TestPluginFactoryService() {}

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {}

  private:
    std::unique_ptr<DummyBase> plugin1;
    std::unique_ptr<DummyBase> plugin2;
  };

  TestPluginFactoryService::TestPluginFactoryService(const edm::ParameterSet& iConfig):
    plugin1(edm::Service<edm::service::PluginFactoryService>()->create<DummyFactory>(iConfig.getParameter<std::string>("plugin1"), 1))
  {
    edm::Service<edm::service::PluginFactoryService> pfs;
    plugin2.reset(pfs->create<DummyFactory>(iConfig.getParameter<std::string>("plugin2"), 2));

    std::string label = iConfig.getParameter<std::string>("@module_label");

    test_equal(plugin1->get(), 1, label, __LINE__);
    test_equal(plugin2->get(), iConfig.getParameter<int>("plugin2ExpectedValue"), label, __LINE__);
  }
}

using edmtest::TestPluginFactoryService;
DEFINE_FWK_MODULE(TestPluginFactoryService);

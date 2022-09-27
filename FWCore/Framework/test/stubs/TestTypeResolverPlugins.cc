#include "FWCore/Framework/test/TestTypeResolvers.h"

namespace edm::test {
  class ConfigurableTestTypeResolverMakerPlugin : public edm::ModuleTypeResolverMaker {
  public:
    ConfigurableTestTypeResolverMakerPlugin(edm::ParameterSet const& pset)
        : variant_(pset.getUntrackedParameter<std::string>("variant")) {
      if (variant_ != "other" and variant_ != "cpu") {
        throw edm::Exception(edm::errors::Configuration) << "Variant must be 'other' or 'cpu'. Got " << variant_;
      }
    }

    std::shared_ptr<ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const& pset) const final {
      std::string variant = variant_;
      if (pset.existsAs<std::string>("variant", false)) {
        variant = pset.getUntrackedParameter<std::string>("variant");
        if (variant != "other" and variant != "cpu") {
          throw edm::Exception(edm::errors::Configuration) << "Variant must be 'other' or 'cpu'. Got " << variant;
        }
      }
      return std::make_shared<ConfigurableTestTypeResolver>(variant);
    }

  private:
    std::string variant_;
  };
}  // namespace edm::test

#include "FWCore/Framework/interface/ModuleTypeResolverMakerFactory.h"
DEFINE_EDM_PLUGIN(edm::ModuleTypeResolverMakerFactory,
                  edm::test::ConfigurableTestTypeResolverMakerPlugin,
                  "edm::test::ConfigurableTestTypeResolverMakerPlugin");

#include "FWCore/Framework/test/TestTypeResolvers.h"

namespace edm::test {
  class ConfigurableTestTypeResolverMakerPlugin : public edm::ModuleTypeResolverMaker {
  public:
    ConfigurableTestTypeResolverMakerPlugin(edm::ParameterSet const& pset,
                                            std::vector<std::string> const& selectedAccelerators)
        : variant_(pset.getUntrackedParameter<std::string>("variant")) {
      if (variant_.empty()) {
        if (std::find(selectedAccelerators.begin(), selectedAccelerators.end(), "other") !=
            selectedAccelerators.end()) {
          variant_ = "other";
        } else if (std::find(selectedAccelerators.begin(), selectedAccelerators.end(), "cpu") !=
                   selectedAccelerators.end()) {
          variant_ = "cpu";
        } else {
          throw edm::Exception(edm::errors::UnavailableAccelerator) << "No 'cpu' or 'other' accelerator available";
        }
      } else if (variant_ != "other" and variant_ != "cpu") {
        throw edm::Exception(edm::errors::Configuration) << "variant can be empty, 'other', or 'cpu'. Got " << variant_;
      }
    }
    std::shared_ptr<ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const& pset) const final {
      return std::make_shared<ConfigurableTestTypeResolver>(variant_);
    }

  private:
    std::string variant_;
  };
}  // namespace edm::test

#include "FWCore/Framework/interface/ModuleTypeResolverMakerFactory.h"
DEFINE_EDM_PLUGIN(edm::ModuleTypeResolverMakerFactory,
                  edm::test::ConfigurableTestTypeResolverMakerPlugin,
                  "edm::test::ConfigurableTestTypeResolverMakerPlugin");

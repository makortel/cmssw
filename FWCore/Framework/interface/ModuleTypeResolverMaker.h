#ifndef FWCore_Framework_ModuleTypeResolverMaker_h
#define FWCore_Framework_ModuleTypeResolverMaker_h

#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"

namespace edm {
  class ParameterSet;

  /**
   * A derived class of this abstract base class is meant to be owned
   * by the EventProcessor (or equivalent). The concrete object can be
   * created with a plugin factory (in case of EventProcessor the
   * plugin name comes from the configuration). The class provides a
   * factory function to create a concrete ModuleTypeResolverBase
   * object based on the module PSet.
   */
  class ModuleTypeResolverMaker {
  public:
    virtual ~ModuleTypeResolverMaker() = default;

    /**
     * This function creates an implementation of the ModuleTypeResolverBase class based on the module PSet.
     *
     * The return value reflects that the implementation
     * ModuleTypeResolverMaker may cache the ModuleTypeResolverBase objects if it
     * wants to.
     */
    virtual std::shared_ptr<ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const& modulePSet) const = 0;
  };
}  // namespace edm

#endif

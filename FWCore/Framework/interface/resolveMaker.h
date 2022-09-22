#ifndef FWCore_Framework_interface_resolveMaker_h
#define FWCore_Framework_interface_resolveMaker_h

#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>

namespace edm::detail {
  void annotateResolverMakerExceptionAndRethrow(cms::Exception& except,
                                                std::string const& modtype,
                                                ModuleTypeResolverBase const* resolver);

  // Returns a tuple of
  // - non-owning pointer to the maker
  // - boolean indicating if insertion to cache was successful
  //  * true: insertion successful, or maker was found from the cache
  //  * false: failed to insert the maker to cache
  template <typename TFactory, typename TCache>
  auto resolveMaker(std::string const& moduleType,
                    ModuleTypeResolverMaker const* resolverMaker,
                    edm::ParameterSet const& modulePSet,
                    TCache& makerCache) -> std::tuple<typename TCache::mapped_type::element_type*, bool> {
    if (resolverMaker) {
      auto resolver = resolverMaker->makeResolver(modulePSet);
      auto index = resolver->kInitialIndex;
      auto newType = moduleType;
      do {
        auto [ttype, tindex] = resolver->resolveType(std::move(newType), index);
        newType = std::move(ttype);
        index = tindex;
        // try the maker cache first
        auto found = makerCache.find(newType);
        if (found != makerCache.end()) {
          return {found->second.get(), true};
        }

        // if not in cache, then try to create
        auto m = TFactory::get()->tryToCreate(newType);
        if (m) {
          //FDEBUG(1) << "Factory:  created worker of type " << modtype << std::endl;
          auto [it, succeeded] = makerCache.emplace(newType, std::move(m));
          if (not succeeded) {
            return {nullptr, false};
          }
          return {it->second.get(), true};
        }
        // not found, try next one
      } while (index != resolver->kLastIndex);
      try {
        //failed to find a plugin
        auto m = TFactory::get()->create(moduleType);
        return {nullptr, false};  // dummy return, the create() call throws an exception
      } catch (cms::Exception& iExcept) {
        detail::annotateResolverMakerExceptionAndRethrow(iExcept, moduleType, resolver.get());
      }
    }
    auto [it, succeeded] = makerCache.emplace(moduleType, TFactory::get()->create(moduleType));
    //FDEBUG(1) << "Factory:  created worker of type " << modtype << std::endl;
    if (not succeeded) {
      return {nullptr, false};
    }
    return {it->second.get(), true};
  }
}  // namespace edm::detail

#endif

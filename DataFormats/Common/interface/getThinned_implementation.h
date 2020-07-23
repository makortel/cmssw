#ifndef DataFormats_Common_getThinned_implementation_h
#define DataFormats_Common_getThinned_implementation_h

#include <optional>
#include <tuple>

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"

namespace edm {
  class WrapperBase;

  namespace detail {
    // This function provides a common implementation of
    // EDProductGetter::getThinnedKeyFrom() for EventPrincipal,
    // DataGetterHelper, and BareRootProductGetter.
    //
    // getThinnedProduct assumes getIt was already called and failed to find
    // the product. The input key is the index of the desired element in the
    // container identified by ProductID (which cannot be found).
    // If the return value is not null, then the desired element was found
    // in a thinned container and key is modified to be the index into
    // that thinned container. If the desired element is not found, then
    // nullptr is returned.
    template <typename F1, typename F2, typename F3>
    std::optional<std::tuple<WrapperBase const*, unsigned int> > getThinnedProduct(
        ProductID const& pid,
        unsigned int key,
        ThinnedAssociationsHelper const& thinnedAssociationsHelper,
        F1&& pidToBid,
        F2&& getThinnedAssociation,
        F3&& getIt) {
      BranchID parent = pidToBid(pid);

      // Loop over thinned containers which were made by selecting elements from the parent container
      for (auto associatedBranches = thinnedAssociationsHelper.parentBegin(parent),
                iEnd = thinnedAssociationsHelper.parentEnd(parent);
           associatedBranches != iEnd;
           ++associatedBranches) {
        ThinnedAssociation const* thinnedAssociation = getThinnedAssociation(associatedBranches->association());
        if (thinnedAssociation == nullptr)
          continue;

        if (associatedBranches->parent() != pidToBid(thinnedAssociation->parentCollectionID())) {
          continue;
        }

        unsigned int thinnedIndex = 0;
        // Does this thinned container have the element referenced by key?
        // If yes, thinnedIndex is set to point to it in the thinned container
        if (!thinnedAssociation->hasParentIndex(key, thinnedIndex)) {
          continue;
        }
        // Get the thinned container and return a pointer if we can find it
        ProductID const& thinnedCollectionPID = thinnedAssociation->thinnedCollectionID();
        WrapperBase const* thinnedCollection = getIt(thinnedCollectionPID);
        if (thinnedCollection == nullptr) {
          // Thinned container is not found, try looking recursively in thinned containers
          // which were made by selecting elements from this thinned container.
          auto thinnedCollectionKey = getThinnedProduct(thinnedCollectionPID,
                                                        thinnedIndex,
                                                        thinnedAssociationsHelper,
                                                        std::forward<F1>(pidToBid),
                                                        std::forward<F2>(getThinnedAssociation),
                                                        std::forward<F3>(getIt));
          if (thinnedCollectionKey.has_value()) {
            return thinnedCollectionKey;
          } else {
            continue;
          }
        }
        return std::tuple(thinnedCollection, thinnedIndex);
      }
      return {};
    }
  }  // namespace detail
}  // namespace edm

#endif

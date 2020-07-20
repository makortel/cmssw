#ifndef DataFormats_Common_getThinnedKeyFrom_implementation_h
#define DataFormats_Common_getThinnedKeyFrom_implementation_h

#include <optional>
#include <vector>

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm::detail {
  // This function provides a common implementation of
  // EDProductGetter::getThinnedKeyFrom() for EventPrincipal,
  // DataGetterHelper, and BareRootProductGetter.
  //
  // The thinned ProductID must come from an existing RefCore. The
  // input key is the index of the desired element in the container
  // identified by the parent ProductID. If the return value is not
  // null, then the desired element was found in a thinned container.
  // If the desired element is not found, then an optional without a
  // value is returned.
  template <typename F>
  std::optional<unsigned int> getThinnedKeyFrom_implementation(
      ProductID const& parentID,
      BranchID const& parent,
      unsigned int key,
      ProductID const& thinnedID,
      BranchID thinned,
      ThinnedAssociationsHelper const& thinnedAssociationsHelper,
      F&& getThinnedAssociation) {
    if (thinnedAssociationsHelper.parentBegin(parent) == thinnedAssociationsHelper.parentEnd(parent)) {
      throw Exception(errors::InvalidReference)
          << "Parent collection with ProductID " << parentID << " has not been thinned";
    }

    bool foundParent = false;
    std::vector<ThinnedAssociation const*> thinnedAssociationParentage;
    while (not foundParent) {
      // TODO: be smarter than linear search every time?
      auto branchesToThinned = std::find_if(
          thinnedAssociationsHelper.begin(), thinnedAssociationsHelper.end(), [&thinned](auto& associatedBranches) {
            return associatedBranches.thinned() == thinned;
          });
      if (branchesToThinned == thinnedAssociationsHelper.end()) {
        if (thinnedAssociationParentage.empty()) {
          throw Exception(errors::ProductNotFound) << "Thinned collection with ProductID " << thinnedID << " not found";
        } else {
          throw Exception(errors::InvalidReference) << "Requested thinned collection with ProductID " << thinnedID
                                                    << " is not thinned from the parent collection with ProductID "
                                                    << parentID << " or from any collection thinned from it.";
        }
      }

      ThinnedAssociation const* thinnedAssociation = getThinnedAssociation(branchesToThinned->association());
      if (thinnedAssociation == nullptr) {
        Exception ex{errors::LogicError};
        if (thinnedAssociationParentage.empty()) {
          ex << "ThinnedAssociation corresponding to thinned collection with ProductID " << thinnedID << " not found.";
        } else {
          ex << "Intermediate ThinnedAssociation between the requested thinned ProductID " << thinnedID
             << " and parent " << parentID << " not found.";
        }
        ex << " This should not happen.\nPlease contact the core framework developers.";
        throw ex;
      }

      thinnedAssociationParentage.push_back(thinnedAssociation);
      if (branchesToThinned->parent() == parent) {
        foundParent = true;
      } else {
        // next iteration with current parent as the thinned collection
        thinned = branchesToThinned->parent();
      }
    }

    // found the parent, now need to rewind the parentage chain to
    // find the index in the requested thinned collection
    unsigned int parentIndex = key;
    unsigned int thinnedIndex = 0;
    for (auto iAssociation = thinnedAssociationParentage.rbegin(), iEnd = thinnedAssociationParentage.rend();
         iAssociation != iEnd;
         ++iAssociation) {
      if ((*iAssociation)->hasParentIndex(parentIndex, thinnedIndex)) {
        parentIndex = thinnedIndex;
      } else {
        return std::nullopt;
      }
    }
    return thinnedIndex;
  }
}  // namespace edm::detail

#endif

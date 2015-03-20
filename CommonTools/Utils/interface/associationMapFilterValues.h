#ifndef CommonTools_Utils_associationMapFilterValues_h
#define CommonTools_Utils_associationMapFilterValues_h

#include "DataFormats/Common/interface/Ref.h"

#include <unordered_set>

namespace associationMapFilterValuesHelpers {
  // Common implementation for Ref and RefToBase
  template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
  void ref_insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
    if(value_indices.find(keyValue.val.key()) != value_indices.end()) {
      ret.insert(keyValue);
    }
  }

  // By default no implementation, as it turns out to be very specific for the types
  template <typename DataTag>
  struct IfFound;

  // Specialize for Ref and RefToBase, implementation is the same
  template <typename C, typename T, typename F>
  struct IfFound<edm::Ref<C, T, F>> {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      ref_insert(ret, keyValue, value_indices);
    }
  };

  template <typename T>
  struct IfFound<edm::RefToBase<T>> {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      ref_insert(ret, keyValue, value_indices);
    }
  };

  //
}

template <typename T_AssociationMap, typename T_RefVector>
T_AssociationMap associationMapFilterValues(const T_AssociationMap& map, const T_RefVector& valueRefs) {
  T_AssociationMap ret;

  // First copy the keys of values to a set for faster lookup of their existence in the map
  std::unordered_set<typename T_AssociationMap::index_type> value_indices;
  for(const auto& ref: valueRefs) {
    value_indices.insert(ref.key());
  }

  for(const auto& keyValue: map) {
    associationMapFilterValuesHelpers::IfFound<typename T_AssociationMap::data_type>::insert(ret, keyValue, value_indices);
    /*
    if(value_indices.find(keyValue.val.key()) != value_indices.end()) {
      ret.insert(keyValue);
    }
    */
  }
    

  return ret;
}

#endif


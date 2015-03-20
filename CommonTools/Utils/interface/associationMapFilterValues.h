#ifndef CommonTools_Utils_associationMapFilterValues_h
#define CommonTools_Utils_associationMapFilterValues_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include <unordered_set>

namespace associationMapFilterValuesHelpers {
  // Common implementation
  template <typename T_AssociationMap, typename T_Key, typename T_Value,
            typename T_ValueIndices>
  void findInsert(T_AssociationMap& ret, const T_Key& key, const T_Value& value,
                     const T_ValueIndices& value_indices) {
    if(value_indices.find(value.key()) != value_indices.end()) {
      ret.insert(key, value);
    }
  }

  // By default no implementation, as it turns out to be very specific for the types
  template <typename ValueTag>
  struct IfFound;

  // Specialize for Ref and RefToBase, implementation is the same
  template <typename C, typename T, typename F>
  struct IfFound<edm::Ref<C, T, F>> {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      findInsert(ret, keyValue.key, keyValue.val, value_indices);
    }
  };

  template <typename T>
  struct IfFound<edm::RefToBase<T>> {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      findInsert(ret, keyValue.key, keyValue.val, value_indices);
    }
  };

  // Specialize for RefVector
  template <typename C, typename T, typename F>
  struct IfFound<edm::RefVector<C, T, F>> {
    template <typename T_AssociationMap, typename T_KeyValue, typename T_ValueIndices>
    static void insert(T_AssociationMap& ret, const T_KeyValue& keyValue, const T_ValueIndices& value_indices) {
      for(const auto& value: keyValue.val) {
        findInsert(ret, keyValue.key, value, value_indices);
      }
    }
  };
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
    associationMapFilterValuesHelpers::IfFound<typename T_AssociationMap::value_type::value_type>::insert(ret, keyValue, value_indices);
  }
    

  return ret;
}

#endif


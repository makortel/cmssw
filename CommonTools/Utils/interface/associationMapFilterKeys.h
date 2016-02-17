#ifndef CommonTools_Utils_associationMapFilterKeys_h
#define CommonTools_Utils_associationMapFilterKeys_h

#include "CommonTools/Utils/interface/associationMapFilterValues.h"

namespace associationMapFilterKeysHelpers {
  // Common implementation
  struct FindValueInsert {
    template <typename T_AssociationMap, typename T_Key, 
              typename T_ValueIndex, typename T_Value,
              typename T_KeyIndices>
    static void call(T_AssociationMap& ret, const T_Key& key,
                     const T_ValueIndex& valueIndex, const T_Value& value,
                     const T_KeyIndices& key_indices) {
      if(key_indices.find(key.key()) != key_indices.end()) {
        ret.insert(key, value);
      }
    }
  };
}

/**
 * Filters entries of AssociationMap by keeping only those
 * associations that have a key in a given collection
 *
 * @param map        AssociationMap to filter
 * @param keyRefs    Collection of Refs to keys.
 *
 * @tparam T_AssociationMap  Type of the AssociationMap
 * @tparam T_RefVector       Type of the Ref collection.
 *
 * For AssociationMap<Tag<CKey, CVal>>, the collection of Refs can be
 * RefVector<CVal>, vector<Ref<CVal>>, vector<RefToBase<CVal>>, or
 * View<T>. More can be supported if needed.
 *
 * @return A filtered copy of the AssociationMap
 *
 * Throws if the keys of AssociationMap and keyRefs point to
 * different collections (similar check as in
 * AssociationMap::operator[] for the keys).
 */
template <typename T_AssociationMap, typename T_RefVector>
T_AssociationMap associationMapFilterKeys(const T_AssociationMap& map, const T_RefVector& keyRefs) {
  // If the input map is empty, just return it in order to avoid an
  // exception from failing edm::helpers::checkRef() (in this case the
  // refProd points to (0,0) that will fail the check).
  if(map.empty())
    return map;

  T_AssociationMap ret(map.refProd());

  // First copy the keys of keys to a set for faster lookup of their existence in the map
  std::unordered_set<typename T_AssociationMap::index_type> key_indices;
  associationMapFilterValuesHelpers::FillIndices<T_RefVector>::fill(key_indices, keyRefs, map.refProd().key);

  for(const auto& keyValue: map) {
    associationMapFilterValuesHelpers::IfFound<typename T_AssociationMap::value_type::value_type,
                                               associationMapFilterKeysHelpers::FindValueInsert>::insert(ret, keyValue, key_indices);
  }
    

  return ret;
}

#endif


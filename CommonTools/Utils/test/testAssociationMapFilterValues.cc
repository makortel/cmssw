#include "CommonTools/Utils/interface/associationMapFilterValues.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include <cppunit/extensions/HelperMacros.h>

class testAssociationMapFilterValues : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testAssociationMapFilterValues);
  CPPUNIT_TEST(checkOneToOne);
  CPPUNIT_TEST(checkOneToMany);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkOneToOne(); 
  void checkOneToMany(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION( testAssociationMapFilterValues );

void testAssociationMapFilterValues::checkOneToOne() {
  typedef std::vector<int> CKey;
  typedef std::vector<double> CVal;
  typedef edm::AssociationMap<edm::OneToOne<CKey, CVal, unsigned char> > Assoc;

  CKey keys{1, 2, 3};
  CVal values{1.0, 2.0, 3.0};

  Assoc map;
  map.insert(edm::Ref<CKey>(&keys, 0), edm::Ref<CVal>(&values, 0));
  map.insert(edm::Ref<CKey>(&keys, 1), edm::Ref<CVal>(&values, 1));
  map.insert(edm::Ref<CKey>(&keys, 2), edm::Ref<CVal>(&values, 2));

  std::vector<edm::Ref<CVal>> keep{edm::Ref<CVal>(&values, 0), edm::Ref<CVal>(&values, 2)};
  Assoc filtered = associationMapFilterValues(map, keep);
  CPPUNIT_ASSERT( filtered.size() == 2 );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 0)) != filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 1)) == filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 2)) != filtered.end() );

  edm::RefVector<CVal> keep2;
  keep2.push_back(edm::Ref<CVal>(&values, 1));
  filtered = associationMapFilterValues(map, keep2);
  CPPUNIT_ASSERT( filtered.size() == 1 );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 0)) == filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 1)) != filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 2)) == filtered.end() );
}

void testAssociationMapFilterValues::checkOneToMany() {
/*
  typedef std::vector<int> CKey;
  typedef std::vector<double> CVal;
  typedef edm::AssociationMap<edm::OneToMany<CKey, CVal, unsigned char> > Assoc;

  CKey keys{1, 2, 3, 3};
  CVal values{1.0, 2.0, 3.0, 4.0};

  Assoc map;
  map.insert(edm::Ref<CKey>(&keys, 0), edm::Ref<CVal>(&values, 0));
  map.insert(edm::Ref<CKey>(&keys, 1), edm::Ref<CVal>(&values, 1));
  map.insert(edm::Ref<CKey>(&keys, 2), edm::Ref<CVal>(&values, 2));
  map.insert(edm::Ref<CKey>(&keys, 3), edm::Ref<CVal>(&values, 3));

  std::vector<edm::Ref<CVal>> keep{edm::Ref<CVal>(&values, 0), edm::Ref<CVal>(&values, 2)};

  Assoc filtered = associationMapFilterValues(map, keep);
  CPPUNIT_ASSERT( filtered.size() == 2 );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 0)) != filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 1)) == filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 2)) != filtered.end() );
*/
}

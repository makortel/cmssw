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

namespace {
  class Base {
  public:
    explicit Base(double v): v_(v) {}
    virtual ~Base() {}

    double value() const { return v_; }
  private:
    double v_;
  };

  class Derived: public Base {
  public:
    explicit Derived(double v): Base(v) {}
    virtual ~Derived() {}
  };
}

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

  // vector of Refs
  std::vector<edm::Ref<CVal>> keep{edm::Ref<CVal>(&values, 0), edm::Ref<CVal>(&values, 2)};
  Assoc filtered = associationMapFilterValues(map, keep);
  CPPUNIT_ASSERT( filtered.size() == 2 );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 0)) != filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 1)) == filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 2)) != filtered.end() );

  // RefVector
  edm::RefVector<CVal> keep2;
  keep2.push_back(edm::Ref<CVal>(&values, 1));
  filtered = associationMapFilterValues(map, keep2);
  CPPUNIT_ASSERT( filtered.size() == 1 );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 0)) == filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 1)) != filtered.end() );
  CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 2)) == filtered.end() );

  // Mostly check that it compiles
  edm::View<CVal> keepView;
  filtered = associationMapFilterValues(map, keepView);
  CPPUNIT_ASSERT( filtered.size() == 0 );
}

void testAssociationMapFilterValues::checkOneToMany() {
  // simple data
  {
    typedef std::vector<int> CKey;
    typedef std::vector<double> CVal;
    typedef edm::AssociationMap<edm::OneToMany<CKey, CVal, unsigned char> > Assoc;

    CKey keys{1, 2, 3};
    CVal values{1.0, 2.0, 3.0, 4.0};

    Assoc map;
    map.insert(edm::Ref<CKey>(&keys, 0), edm::Ref<CVal>(&values, 0));
    map.insert(edm::Ref<CKey>(&keys, 1), edm::Ref<CVal>(&values, 1));
    map.insert(edm::Ref<CKey>(&keys, 2), edm::Ref<CVal>(&values, 2));
    map.insert(edm::Ref<CKey>(&keys, 2), edm::Ref<CVal>(&values, 3));

    std::vector<edm::Ref<CVal>> keep{edm::Ref<CVal>(&values, 0), edm::Ref<CVal>(&values, 2)};

    Assoc filtered = associationMapFilterValues(map, keep);
    CPPUNIT_ASSERT( filtered.size() == 2 );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 0)) != filtered.end() );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 1)) == filtered.end() );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 2)) != filtered.end() );
  }

  // Ref data
  {
    typedef std::vector<int> CKey;
    typedef std::vector<double> Data;
    typedef std::vector<edm::Ref<Data>> CVal;
    typedef edm::AssociationMap<edm::OneToMany<CKey, CVal, unsigned char> > Assoc;

    CKey keys{1, 2, 3};
    Data data{1.0, 2.0, 3.0, 4.0};
    CVal values{edm::Ref<Data>(&data, 0), edm::Ref<Data>(&data, 1), edm::Ref<Data>(&data, 2)};

    Assoc map;
    map.insert(edm::Ref<CKey>(&keys, 0), edm::Ref<CVal>(&values, 0));
    map.insert(edm::Ref<CKey>(&keys, 1), edm::Ref<CVal>(&values, 1));
    map.insert(edm::Ref<CKey>(&keys, 2), edm::Ref<CVal>(&values, 1));
    map.insert(edm::Ref<CKey>(&keys, 2), edm::Ref<CVal>(&values, 0));
    map.insert(edm::Ref<CKey>(&keys, 3), edm::Ref<CVal>(&values, 2));

    std::vector<edm::Ref<CVal>> keep{edm::Ref<CVal>(&values, 0), edm::Ref<CVal>(&values, 2)};

    Assoc filtered = associationMapFilterValues(map, keep);
    CPPUNIT_ASSERT( filtered.size() == 3 );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 0)) != filtered.end() );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 1)) == filtered.end() );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 2)) != filtered.end() );
  }

  // RefToBase data
  {
    typedef std::vector<int> CKey;
    typedef std::vector<Derived> Data;
    typedef std::vector<edm::RefToBase<Base>> CVal;
    typedef edm::AssociationMap<edm::OneToMany<CKey, CVal, unsigned char> > Assoc;

    CKey keys{1, 2, 3};
    Data data{Derived(1.0), Derived(2.0), Derived(3.0), Derived(4.0)};

    CVal values{
      edm::RefToBase<Base>(edm::Ref<Data>(&data, 0)),
      edm::RefToBase<Base>(edm::Ref<Data>(&data, 1)),
      edm::RefToBase<Base>(edm::Ref<Data>(&data, 2))
    };

    Assoc map;
    map.insert(edm::Ref<CKey>(&keys, 0), edm::Ref<CVal>(&values, 0));
    map.insert(edm::Ref<CKey>(&keys, 1), edm::Ref<CVal>(&values, 1));
    map.insert(edm::Ref<CKey>(&keys, 2), edm::Ref<CVal>(&values, 1));
    map.insert(edm::Ref<CKey>(&keys, 2), edm::Ref<CVal>(&values, 0));
    map.insert(edm::Ref<CKey>(&keys, 3), edm::Ref<CVal>(&values, 2));

    std::vector<edm::RefToBase<Base>> keep{
      edm::RefToBase<Base>(edm::Ref<Data>(&data, 0)),
      edm::RefToBase<Base>(edm::Ref<Data>(&data, 2))
    };

    Assoc filtered = associationMapFilterValues(map, keep);
    CPPUNIT_ASSERT( filtered.size() == 3 );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 0)) != filtered.end() );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 1)) == filtered.end() );
    CPPUNIT_ASSERT( filtered.find(edm::Ref<CKey>(&keys, 2)) != filtered.end() );
  }
}

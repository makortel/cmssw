#include "RecoPixelVertexing/PixelTriplets/plugins/KDTree.h"

#include <cppunit/extensions/HelperMacros.h>

#include <iostream>

class testKDTree: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testKDTree);

  CPPUNIT_TEST(test2D);
  CPPUNIT_TEST(test3D);
  CPPUNIT_TEST(test4D);
  CPPUNIT_TEST(test5D);
  CPPUNIT_TEST(test6D);
  CPPUNIT_TEST(test7D);
  CPPUNIT_TEST(test8D);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void test2D();
  void test3D();

  template <size_t DIM>
  void testND();

  void test4D() { testND<4>(); }
  void test5D() { testND<5>(); }
  void test6D() { testND<6>(); }
  void test7D() { testND<7>(); }
  void test8D() { testND<8>(); }

};

CPPUNIT_TEST_SUITE_REGISTRATION(testKDTree);

namespace {
  template <typename T, size_t DIM>
  void simpleSearch(const std::vector<KDTreeNodeInfo<T, DIM> >& nodes, const KDTreeBox<DIM>& searchBox, std::vector<KDTreeNodeInfo<T, DIM> >& output) {
    for(const auto& node: nodes) {
      bool inside = true;
      for(size_t i=0; i<DIM; ++i) {
        inside = inside && node.dim[i] >= std::get<0>(searchBox.dims[i]) && node.dim[i] <= std::get<1>(searchBox.dims[i]);
      }
      if(inside) {
        output.push_back(node);
      }
    }
  }

  template <typename T, size_t DIM>
  bool compareResults(const std::vector<T>& kdtreeResult, const std::vector<KDTreeNodeInfo<T, DIM> >& simpleResult, const KDTreeBox<DIM>& searchBox) {
    std::vector<bool> kdtreeExists(kdtreeResult.size(), false);
    std::vector<bool> simpleExists(simpleResult.size(), false);
    size_t sameCount = 0;

    for(size_t i=0; i<kdtreeResult.size(); ++i) {
      for(size_t j=0; j<simpleResult.size(); ++j) {
        if(kdtreeResult[i] == simpleResult[j].data) {
          kdtreeExists[i] = true;
          simpleExists[j] = true;
          ++sameCount;
        }
      }
    }
    if(kdtreeResult.size() == simpleResult.size() && sameCount == kdtreeResult.size())
      return true;

    // print some diagnostic
    std::cout << std::endl
              << "search box";
    for(size_t i=0; i<DIM; ++i) {
      std::cout << " [" << std::get<0>(searchBox.dims[i]) << ", " << std::get<1>(searchBox.dims[i]) << "]";
      if(i != DIM-1)
        std::cout << " x";
    }
    std::cout << std::endl
              << "kdtreeResult.size() " << kdtreeResult.size() << " simpleResult.size() " << simpleResult.size() << " sameCount " << sameCount << std::endl
              << "only in kdtree:";
    for(size_t i=0; i<kdtreeExists.size(); ++i) {
      if(!kdtreeExists[i]) {
        std::cout << " " << kdtreeResult[i];
      }
    }
    std::cout << std::endl << "only in simple:";
    for(size_t i=0; i<simpleExists.size(); ++i) {
      if(!simpleExists[i]) {
        std::cout << " " << simpleResult[i].data;
      }
    }
    std::cout <<  std::endl;

    return false;
  }
}


void testKDTree::test2D() {
  // build the nodes
  std::vector<KDTreeNodeInfo<int> > nodes;
  nodes.reserve(10000);
  size_t id=0;
  for(size_t i=0; i<100; ++i) {
    for(size_t j=0; j<100; ++j) {
      nodes.emplace_back(id, i*0.1f+0.05f, j*0.1f+0.05f);
      /*
      if(nodes.back().dim[0] <= 1.0 && nodes.back().dim[1] <= 1.0)
        std::cout << "Node " << nodes.back().data << " dim0 " << nodes.back().dim[0] << " dim1 " << nodes.back().dim[1] << std::endl;
      */
      ++id;
    }
  }

  // build the tree
  KDTree<int> kdtree;
  CPPUNIT_ASSERT(kdtree.empty());
  kdtree.build(nodes);

  // search in a window around all items, compare result to simple
  std::vector<int> foundNodes;
  std::vector<KDTreeNodeInfo<int> > foundNodes2;
  auto runTest = [&](float window, float offset) {
    for(const auto& node: nodes) {
      KDTreeBox<> box(node.dim[0]-window+offset, node.dim[0]+window+offset, node.dim[1]-window+offset, node.dim[1]+window+offset);
      kdtree.search(box, foundNodes);
      simpleSearch(nodes, box, foundNodes2);
      CPPUNIT_ASSERT(compareResults(foundNodes, foundNodes2, box));
      foundNodes.clear();
      foundNodes2.clear();
    }
  };

  runTest(0.05, 0);
  runTest(0.1, 0);
  runTest(0.15, 0);
  runTest(0.5, 0);

  runTest(0.1, -0.05);
  runTest(0.1, 0.05);
  runTest(0.1, -0.1);
  runTest(0.1, 0.1);

  // do one manual search test
  kdtree.search(KDTreeBox<>(0,1, 0,1), foundNodes);
  CPPUNIT_ASSERT(foundNodes.size() == 100);
  auto myFind = [&](int value) {
    return std::find(foundNodes.begin(), foundNodes.end(), value) != foundNodes.end();
  };
  for(int i=0; i<10; ++i) {
    for(int j=0; j<10; ++j) {
      CPPUNIT_ASSERT(myFind(j*100 + i));
    }
  }

  // clear
  kdtree.clear();
  CPPUNIT_ASSERT(kdtree.empty());
}

void testKDTree::test3D() {
  // build the nodes
  std::vector<KDTreeNodeInfo<int, 3> > nodes;
  nodes.reserve(8000);
  size_t id=0;
  for(size_t i=0; i<20; ++i) {
    for(size_t j=0; j<20; ++j) {
      for(size_t k=0; k<20; ++k) {
        nodes.emplace_back(id, i*0.1f+0.05f, j*0.1f+0.05f, k*0.1f+0.05f);
        /*
        if(nodes.back().dim[0] <= 1.0 && nodes.back().dim[1] <= 1.0 && nodes.back().dim[2] <= 1.0)
          std::cout << "Node " << nodes.back().data << " dim0 " << nodes.back().dim[0] << " dim1 " << nodes.back().dim[1] << " dim2 " << nodes.back().dim[2] << std::endl;
        */
        ++id;
      }
    }
  }

  // build the tree
  KDTree<int, 3> kdtree;
  CPPUNIT_ASSERT(kdtree.empty());
  kdtree.build(nodes);

  // search in a window around all items, compare result to simple
  std::vector<int> foundNodes;
  std::vector<KDTreeNodeInfo<int, 3> > foundNodes2;
  auto runTest = [&](float window, float offset) {
    for(const auto& node: nodes) {
      KDTreeBox<3> box(std::make_tuple(node.dim[0]-window+offset, node.dim[0]+window+offset),
                       std::make_tuple(node.dim[1]-window+offset, node.dim[1]+window+offset),
                       std::make_tuple(node.dim[2]-window+offset, node.dim[2]+window+offset));
      kdtree.search(box, foundNodes);
      simpleSearch(nodes, box, foundNodes2);
      CPPUNIT_ASSERT(compareResults(foundNodes, foundNodes2, box));
      foundNodes.clear();
      foundNodes2.clear();
    }
  };

  runTest(0.05, 0);
  runTest(0.1, 0);

  runTest(0.1, -0.05);
  runTest(0.1, 0.1);

  // do one manual search test
  kdtree.search(KDTreeBox<3>(0,1, 0,1, 0,1), foundNodes);
  CPPUNIT_ASSERT(foundNodes.size() == 1000);
  auto myFind = [&](int value) {
    return std::find(foundNodes.begin(), foundNodes.end(), value) != foundNodes.end();
  };
  for(int i=0; i<10; ++i) {
    for(int j=0; j<10; ++j) {
      for(int k=0; k<10; ++k) {
        //std::cout << "i " << i << " j " << j << " k " << k;
        CPPUNIT_ASSERT(myFind(k*20*20 + j*20 + i));
      }
    }
  }

  // clear
  kdtree.clear();
  CPPUNIT_ASSERT(kdtree.empty());
}

namespace {
  template <size_t DIM>
  struct testTraits {
  };

  template <> struct testTraits<4> {
    const static size_t items_per_dim = 10;
    const static size_t n_items = 10000;
  };
  template <> struct testTraits<5> {
    const static size_t items_per_dim = 6;
    const static size_t n_items = 7776;
  };
  template <> struct testTraits<6> {
    const static size_t items_per_dim = 5;
    const static size_t n_items = 15625;
  };
  template <> struct testTraits<7> {
    const static size_t items_per_dim = 4;
    const static size_t n_items = 16384;
  };
  template <> struct testTraits<8> {
    const static size_t items_per_dim = 3;
    const static size_t n_items = 6561;
  };
}

template <size_t DIM>
void testKDTree::testND() {
  //std::cout << "testing " << DIM << " dimensions" << std::endl;

  // build the nodes
  std::vector<KDTreeNodeInfo<int, DIM> > nodes;
  nodes.reserve(testTraits<DIM>::n_items);
  std::array<size_t, DIM> coordinates{};
  for(size_t id=0; id<testTraits<DIM>::n_items; ++id) {
    std::array<float, DIM> coords;
    //std::cout << "Node " << id;
    for(size_t i=0; i<DIM; ++i) {
      coords[i] = coordinates[i]*0.1f+0.05f;
      //std::cout << " dim" << i << " " << coords[i];
    }
    //std::cout << std::endl;
    nodes.emplace_back(id, coords);

    for(size_t i=0; i<DIM; ++i) {
      coordinates[i] += 1;
      if(coordinates[i] >= testTraits<DIM>::items_per_dim) {
        coordinates[i] = 0;
      }
      else {
        break;
      }
    }
  }

  // build the tree
  KDTree<int, DIM> kdtree;
  CPPUNIT_ASSERT(kdtree.empty());
  kdtree.build(nodes);

  // search in a window around all items, compare result to simple
  std::vector<int> foundNodes;
  std::vector<KDTreeNodeInfo<int, DIM> > foundNodes2;
  auto runTest = [&](float window, float offset) {
    //std::cout << "Running test " << window << " " << offset << std::endl;
    for(const auto& node: nodes) {
      //std::cout << "ID " << node.data << std::endl;
      std::array<std::tuple<float, float>, DIM> boxDims;
      for(size_t i=0; i<DIM; ++i) {
        boxDims[i] = std::make_tuple(node.dim[i]-window+offset, node.dim[i]+window+offset);
      }
      KDTreeBox<DIM> box(boxDims);
      kdtree.search(box, foundNodes);
      simpleSearch(nodes, box, foundNodes2);
      //std::cout << "ID " << node.data << " found nodes " << foundNodes.size() << std::endl;
      CPPUNIT_ASSERT(compareResults(foundNodes, foundNodes2, box));
      foundNodes.clear();
      foundNodes2.clear();
    }
  };

  runTest(0.05, 0);
  runTest(0.1, 0);

  runTest(0.1, -0.05);
  runTest(0.1, 0.1);
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

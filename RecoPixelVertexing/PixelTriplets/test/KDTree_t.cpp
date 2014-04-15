#include "RecoPixelVertexing/PixelTriplets/plugins/KDTree.h"

#include <cppunit/extensions/HelperMacros.h>

#include <iostream>

class testKDTree: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testKDTree);

  CPPUNIT_TEST(test);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testKDTree);

namespace {
  template <typename T>
  void simpleSearch(const std::vector<KDTreeNodeInfo<T> >& nodes, const KDTreeBox& searchBox, std::vector<KDTreeNodeInfo<T> >& output) {
    for(const auto& node: nodes) {
      if(node.dim[0] >= searchBox.dim1min && node.dim[0] <= searchBox.dim1max &&
         node.dim[1] >= searchBox.dim2min && node.dim[1] <= searchBox.dim2max) {
        output.push_back(node);
      }
    }
  }

  template <typename T>
  bool compareResults(const std::vector<T>& kdtreeResult, const std::vector<KDTreeNodeInfo<T> >& simpleResult, const KDTreeBox& searchBox) {
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
              << "search box [" << searchBox.dim1min << ", " << searchBox.dim1max << "] x [" << searchBox.dim2min << ", " << searchBox.dim2max << "]" << std::endl
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


void testKDTree::test() {
  // build the nodes
  std::vector<KDTreeNodeInfo<int> > nodes;
  nodes.reserve(10000);
  size_t id=0;
  for(size_t i=0; i<100; ++i) {
    for(size_t j=0; j<100; ++j) {
      nodes.emplace_back(id, i*0.1+0.05, j*0.1+0.05);
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
  CPPUNIT_ASSERT(kdtree.size() == static_cast<int>(2*nodes.size()-1));

  // search in a window around all items, compare result to simple
  std::vector<int> foundNodes;
  std::vector<KDTreeNodeInfo<int> > foundNodes2;
  auto runTest = [&](float window, float offset) {
    for(const auto& node: nodes) {
      KDTreeBox box(node.dim[0]-window+offset, node.dim[0]+window+offset, node.dim[1]-window+offset, node.dim[1]+window+offset);
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
  kdtree.search(KDTreeBox(0,1, 0,1), foundNodes);
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

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

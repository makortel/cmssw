#ifndef KDTreeLinkerAlgoTemplated_h
#define KDTreeLinkerAlgoTemplated_h

#include "KDTreeTools.h"

#include <cassert>
#include <vector>

#include <iostream>

// Class that implements the KDTree partition of DIM-dimensional space and 
// a closest point search algorithme.

template <typename DATA, size_t DIM=2>
class KDTree
{
 public:
  KDTree() {}
  ~KDTree() {}
  
  // Build the k-d tree from nodeList
  void build(std::vector<KDTreeNodeInfo<DATA, DIM> > &nodeList);
  
  // Search for all items contained in the given rectangular
  // searchBox. The found data elements are stored in resRecHitList.
  void search(const KDTreeBox<DIM>& searchBox,
              std::vector<DATA>& resRecHitList) const;

  // Return true if the tree is empty
  bool empty() const {return nodePool_.empty();}

  // This returns the number of nodes + leaves in the tree
  // (nElements should be (size() +1)/2)
  int size() const { return nodePool_.size();}

  // This method clears all allocated structures.
  void clear() {   nodePool_.clear(); }
  
 private:
  // The node pool allow us to do just 1 call to new for each tree building.
  KDTreeNodes<DATA, DIM> nodePool_;

  //Fast median search with Wirth algorithm in initialList between low and high indexes.
  int medianSearch(const int low,
                   const int high,
                   const int treeDepth,
                   std::vector<KDTreeNodeInfo<DATA, DIM> >& initialList) const;

  // Recursive builder. Is called by build()
  int recBuild(const int low,
               const int hight,
               int depth,
               std::vector<KDTreeNodeInfo<DATA, DIM> >& initialList);

  // Recursive search. Is called by search()
  void recSearch(int current,
                 int depth,
                 const std::array<std::tuple<float, float>, DIM>& dimOthersLimits,
                 std::vector<DATA>& output) const;
};


//Implementation

template <typename DATA, size_t DIM>
void
KDTree<DATA, DIM>::build(std::vector<KDTreeNodeInfo<DATA, DIM> > &nodeList)
{
  if(!nodeList.empty()) {
    size_t size = nodeList.size();
    nodePool_.build(size);
    
    // Here we build the KDTree
    int root = recBuild(0, size, 0, nodeList);
    assert(root == 0);
  }
}
 
template <typename DATA, size_t DIM>
int
KDTree<DATA, DIM>::medianSearch(const int low,
                                const int high,
                                const int treeDepth,
                                std::vector<KDTreeNodeInfo<DATA, DIM> >& initialList) const
{
  int nbrElts = high - low;
  int median = (nbrElts & 1)	? nbrElts / 2 
				: nbrElts / 2 - 1;
  median += low;

  int l = low;
  int m = high - 1;

  const int dimIndex = treeDepth % DIM;

  while (l < m) {
    KDTreeNodeInfo<DATA, DIM> item = initialList[median];
    int i = l;
    int j = m;

    do {
      kdtreetraits::rewindIndices(initialList, item, i, j, dimIndex);

      if (i <= j){
	std::swap(initialList[i], initialList[j]);
	i++; 
	j--;
      }
    } while (i <= j);
    if (j < median) l = i;
    if (i > median) m = j;
  }

  return median;
}



template <typename DATA, size_t DIM>
void
KDTree<DATA, DIM>::search(const KDTreeBox<DIM>& trackBox,
                          std::vector<DATA>& recHits) const
{
  if (!empty()) {
    recSearch(0, 0, trackBox.dims, recHits);
  }
}


template <typename DATA, size_t DIM>
void 
KDTree<DATA, DIM>::recSearch(int current, int depth,
                             const std::array<std::tuple<float, float>, DIM>& dimLimits,
                             std::vector<DATA>& output) const
{
  // Iterate until leaf is found, or there are no children in the
  // search window. If search has to proceed on both children, proceed
  // the search to left child via recursion. Swap search window
  // dimension on alternate levels.
  while(true) {
    const int right = nodePool_.right[current];
    const int dimIndex = depth % DIM;
    const float median = nodePool_.dimensions[current][dimIndex];

    const bool isLeaf = nodePool_.isLeaf(right);
    bool goLeft = (std::get<0>(dimLimits[dimIndex]) <= median);
    bool goRight = (std::get<1>(dimLimits[dimIndex]) >= median);

    //std::cout << "current " << current << " right " << right << " depth " << depth << " goLeft " << goLeft << " goRight " << goRight << std::endl;

    if(goLeft & goRight) {
      // If point inside the rectangle/area
      // Use intentionally bit-wise & instead of logical && for better
      // performance. It is faster to always do all comparisons than to
      // allow use of branches to not do some if any of the first ones
      // is false.
      /*
      const float other = nodePool_.dimensions[current][1-dimIndex];
      if((std::get<0>(dimLimits[1-dimIndex]) <= other) & (std::get<1>(dimLimits[1-dimIndex]) >= other)) {
      */
      /*
      bool inside = true;
      for(size_t i=0; i<DIM; ++i) {
        //if(i == dimIndex) continue;
        //if(!inside) break;
        const float other = nodePool_.dimensions[current][i];
        const std::tuple<float, float>& limits = dimLimits[i];
        inside = inside & (std::get<0>(limits) <= other) & (std::get<1>(limits) >= other);
      }
      if(inside) {
      */
      if(kdtreetraits::isInside(nodePool_.dimensions[current], dimLimits, dimIndex)) {
        output.push_back(nodePool_.data[current]);
      }
    }
    if(isLeaf)
      break;
    if(nodePool_.hasOneDaughter(right)) {
      ++current;
      continue;
    }

    ++depth;
    if(goLeft & goRight) {
      const int left = current+1;
      recSearch(left, depth, dimLimits, output);
      // continue with right
      current = right;
    }
    else if(goLeft) {
      ++current;
    }
    else if(goRight) {
      current = right;
    }
    else {
      break;
    }
  }
}

template <typename DATA, size_t DIM>
int
KDTree<DATA, DIM>::recBuild(const int low,
                            const int high,
                            int depth,
                            std::vector<KDTreeNodeInfo<DATA, DIM> >& initialList)
{
  const bool isLeaf = (low+1 == high);
  const int nodeInd = nodePool_.getNextNode();
  const int medianId = isLeaf ? low : medianSearch(low, high, depth, initialList);

  const KDTreeNodeInfo<DATA, DIM>& info = initialList[medianId];
  nodePool_.data[nodeInd] = info.data;
  nodePool_.dimensions[nodeInd] = info.dim;

  //std::cout << "depth " << depth << " low " << low << " high " << high << " medianId " << medianId << std::endl;

  if(isLeaf) {
    nodePool_.right[nodeInd] = 0;
  }
  else {
    // We recursively build the son nodes
    ++depth;
    int ndaughters = 0;
    if(medianId > low) {
      int left = recBuild(low, medianId, depth, initialList);
      assert(nodeInd+1 == left);
      ndaughters = 1;
    }
    if(high > medianId) {
      int right = recBuild(medianId+1, high, depth, initialList);
      if(ndaughters == 0) {
        assert(nodeInd+1 == right);
        ndaughters = 1;
      }
      else {
        ndaughters = right;
      }
      
    }

    nodePool_.right[nodeInd] = ndaughters;
  }

  return nodeInd;
}

#endif

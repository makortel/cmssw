#ifndef KDTreeLinkerAlgoTemplated_h
#define KDTreeLinkerAlgoTemplated_h

#include "KDTreeTools.h"

#include <cassert>
#include <vector>

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
                 std::tuple<float, float> dimCurrLimits,
                 std::array<std::tuple<float, float>, DIM-1> dimOthersLimits,
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
      KDTreeTraits<DATA, DIM>::rewindIndices(initialList, item, i, j, dimIndex);

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
    std::array<std::tuple<float, float>, DIM-1> otherDims;
    std::copy(trackBox.dims.begin()+1, trackBox.dims.end(), otherDims.begin());
    recSearch(0, 0, trackBox.dims[0], otherDims, recHits);
  }
}


template <typename DATA, size_t DIM>
void 
KDTree<DATA, DIM>::recSearch(int current, int depth,
                             std::tuple<float, float> dimCurrLimits,
                             std::array<std::tuple<float, float>, DIM-1> dimOthersLimits,
                             std::vector<DATA>& output) const
{
  // Iterate until leaf is found, or there are no children in the
  // search window. If search has to proceed on both children, proceed
  // the search to left child via recursion. Swap search window
  // dimension on alternate levels.
  while(true) {
    int right = nodePool_.right[current];
    if(nodePool_.isLeaf(right)) {
      float dimCurr = nodePool_.median[current];

      // If point inside the rectangle/area
      // Use intentionally bit-wise & instead of logical && for better
      // performance. It is faster to always do all comparisons than to
      // allow use of branches to not do some if any of the first ones
      // is false.
      if((dimCurr >= std::get<0>(dimCurrLimits)) & (dimCurr <= std::get<1>(dimCurrLimits))) {
        std::array<float, DIM-1> dimOthers = nodePool_.dimOthers[current];
        if(KDTreeTraits<DATA, DIM>::isInside(dimOthers, dimOthersLimits)) {
          output.push_back(nodePool_.data[current]);
        }
      }
      break;
    }
    else {
      float median = nodePool_.median[current];

      const bool goLeft = (std::get<0>(dimCurrLimits) <= median);
      const bool goRight = (std::get<1>(dimCurrLimits) >= median);

      // Swap dimension for the next search level
      KDTreeTraits<DATA, DIM>::swapLimits(depth, dimCurrLimits, dimOthersLimits);
      ++depth;
      if(goLeft & goRight) {
        int left = current+1;
        recSearch(left, depth, dimCurrLimits, dimOthersLimits, output);
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
}

template <typename DATA, size_t DIM>
int
KDTree<DATA, DIM>::recBuild(const int low,
                            const int high,
                            int depth,
                            std::vector<KDTreeNodeInfo<DATA, DIM> >& initialList)
{
  const int portionSize = high - low;
  const int dimIndex = depth % DIM;

  if (portionSize == 1) { // Leaf case
    const int leaf = nodePool_.getNextNode();
    const KDTreeNodeInfo<DATA, DIM>& info = initialList[low];
    nodePool_.right[leaf] = 0;
    nodePool_.median[leaf] = info.dim[dimIndex]; // dimCurrent
    const int otherStart = (depth/DIM) % (DIM-1); // there is peculiar rotation happening
    for(size_t i=0, j=otherStart; i<DIM; ++i) {
      if(i != dimIndex) { // skip the "current" dimension
        nodePool_.dimOthers[leaf][j] = info.dim[i];
        j = (j+1) % (DIM-1);
      }
    }
    nodePool_.data[leaf] = info.data;
    return leaf;

  } else { // Node case
    
    // The even depth is associated to dim1 dimension
    // The odd one to dim2 dimension
    int medianId = medianSearch(low, high, depth, initialList);
    const float medianVal = initialList[medianId].dim[dimIndex];

    // We create the node
    const int nodeInd = nodePool_.getNextNode();
    nodePool_.median[nodeInd] = medianVal;

    ++depth;
    ++medianId;

    // We recursively build the son nodes
    int left = recBuild(low, medianId, depth, initialList);
    assert(nodeInd+1 == left);
    nodePool_.right[nodeInd] = recBuild(medianId, high, depth, initialList);

    return nodeInd;
  }
}

#endif

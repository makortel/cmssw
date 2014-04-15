#ifndef KDTreeLinkerAlgoTemplated_h
#define KDTreeLinkerAlgoTemplated_h

#include "KDTreeTools.h"

#include <cassert>
#include <vector>

// Class that implements the KDTree partition of 2D space and 
// a closest point search algorithme.

template <typename DATA>
class KDTree
{
 public:
  KDTree();
  ~KDTree();
  
  // Build the k-d tree from nodeList
  void build(std::vector<KDTreeNodeInfo<DATA> > &nodeList);
  
  // Search for all items contained in the given rectangular
  // searchBox. The found data elements are stored in resRecHitList.
  void search(const KDTreeBox& searchBox,
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
  KDTreeNodes<DATA> nodePool_;

  //Fast median search with Wirth algorithm in initialList between low and high indexes.
  int medianSearch(const int low,
                   const int high,
                   const int treeDepth,
                   std::vector<KDTreeNodeInfo<DATA> >& initialList) const;

  // Recursive builder. Is called by build()
  int recBuild(const int low,
               const int hight,
               int depth,
               std::vector<KDTreeNodeInfo<DATA> >& initialList);

  // Recursive search. Is called by search()
  void recSearch(int current,
                 float dimCurrMin, float dimCurrMax,
                 float dimOtherMin, float dimOtherMax,
                 std::vector<DATA>& output) const;
};


//Implementation

template < typename DATA >
void
KDTree<DATA>::build(std::vector<KDTreeNodeInfo<DATA> > &nodeList)
{
  if(!nodeList.empty()) {
    size_t size = nodeList.size();
    nodePool_.build(size);
    
    // Here we build the KDTree
    int root = recBuild(0, size, 0, nodeList);
    assert(root == 0);
  }
}
 
template < typename DATA >
int
KDTree<DATA>::medianSearch(const int low,
                           const int high,
                           const int treeDepth,
                           std::vector<KDTreeNodeInfo<DATA> >& initialList) const
{
  int nbrElts = high - low;
  int median = (nbrElts & 1)	? nbrElts / 2 
				: nbrElts / 2 - 1;
  median += low;

  int l = low;
  int m = high - 1;
  
  while (l < m) {
    KDTreeNodeInfo<DATA> elt = initialList[median];
    int i = l;
    int j = m;

    do {
      // The even depth is associated to dim1 dimension
      // The odd one to dim2 dimension
      if (treeDepth & 1) {
	while (initialList[i].dim[1] < elt.dim[1]) i++;
	while (initialList[j].dim[1] > elt.dim[1]) j--;
      } else {
	while (initialList[i].dim[0] < elt.dim[0]) i++;
	while (initialList[j].dim[0] > elt.dim[0]) j--;
      }

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



template < typename DATA >
void
KDTree<DATA>::search(const KDTreeBox& trackBox,
                     std::vector<DATA>& recHits) const
{
  if (!empty()) {
    recSearch(0, trackBox.dim1min, trackBox.dim1max, trackBox.dim2min, trackBox.dim2max, recHits);
  }
}


template < typename DATA >
void 
KDTree<DATA>::recSearch(int current,
                        float dimCurrMin, float dimCurrMax,
                        float dimOtherMin, float dimOtherMax,
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
      if((dimCurr >= dimCurrMin) & (dimCurr <= dimCurrMax)) {
        float dimOther = nodePool_.dimOther[current];
        if((dimOther >= dimOtherMin) & (dimOther <= dimOtherMax)) {
          output.push_back(nodePool_.data[current]);
        }
      }
      break;
    }
    else {
      float median = nodePool_.median[current];

      bool goLeft = (dimCurrMin <= median);
      bool goRight = (dimCurrMax >= median);

      // Swap dimension for the next search level
      std::swap(dimCurrMin, dimOtherMin);
      std::swap(dimCurrMax, dimOtherMax);
      if(goLeft & goRight) {
        int left = current+1;
        recSearch(left, dimCurrMin, dimCurrMax, dimOtherMin, dimOtherMax, output);
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

template <typename DATA>
KDTree<DATA>::KDTree()
{
}

template <typename DATA>
KDTree<DATA>::~KDTree()
{
}

template <typename DATA>
int
KDTree<DATA>::recBuild(const int low,
                       const int high,
                       int depth,
                       std::vector<KDTreeNodeInfo<DATA> >& initialList)
{
  const int portionSize = high - low;
  const int dimIndex = depth&1;

  if (portionSize == 1) { // Leaf case
    const int leaf = nodePool_.getNextNode();
    const KDTreeNodeInfo<DATA>& info = initialList[low];
    nodePool_.right[leaf] = 0;
    nodePool_.median[leaf] = info.dim[dimIndex]; // dimCurrent
    nodePool_.dimOther[leaf] = info.dim[1-dimIndex];
    nodePool_.data[leaf] = info.data;
    return leaf;

  } else { // Node case
    
    // The even depth is associated to dim1 dimension
    // The odd one to dim2 dimension
    int medianId = medianSearch(low, high, depth, initialList);
    float medianVal = initialList[medianId].dim[dimIndex];

    // We create the node
    int nodeInd = nodePool_.getNextNode();
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

#ifndef KDTreeLinkerToolsTemplated_h
#define KDTreeLinkerToolsTemplated_h

#include <algorithm>
#include <array>
#include <tuple>

// Box structure used to define DIM-dimensional field.
// It's used in KDTree building step to divide the detector
// space (ECAL, HCAL...) and in searching step to create a bounding
// box around the demanded point (Track collision point, PS projection...).
template <size_t DIM=2>
struct KDTreeBox
{
  using ArrayType = std::array<std::tuple<float, float>, DIM>; // min, max in tuple
  ArrayType dims;

  // by default initialize to 0
  KDTreeBox(): dims{} {}

  // Construct from a list of tuple arguments
  template <typename ...Args>
  KDTreeBox(Args&&...args)
  {
    fill<0>(dims, std::forward<Args>(args)...);
  }

  KDTreeBox(ArrayType dim): dims(dim) {}

private:
  // Recursive filler from a list of tuples
  template <size_t INDEX, typename ...Args>
  constexpr void fill(ArrayType& arr, std::tuple<float, float> d, Args&&...args) {
    arr[INDEX] = d;
    fill<INDEX+1>(arr, std::forward<Args>(args)...);
  }
  // Recursive filler from a list of floats
  template <size_t INDEX, typename ...Args>
  constexpr void fill(ArrayType& arr, float dmin, float dmax, Args&&...args) {
    arr[INDEX] = std::make_tuple(dmin, dmax);
    fill<INDEX+1>(arr, std::forward<Args>(args)...);
  }

  // Terminate the recursion
  template <size_t INDEX>
  constexpr void fill(ArrayType& arr) {
    static_assert(INDEX <= DIM, "Got more arguments than the dimension DIM");
  }
};

  
// Data stored in each KDTree node.
template <typename DATA, size_t DIM=2>
struct KDTreeNodeInfo 
{
  DATA data;
  std::array<float, DIM> dim;

  public:
  KDTreeNodeInfo()
  {}

  // Construct from an array of coordinates
  KDTreeNodeInfo(const DATA& d, std::array<float, DIM> dim_):
    data(d), dim(dim_)
  {}

  // Construct from a list of coordinate arguments
  template <typename ...Args>
  KDTreeNodeInfo(const DATA& d, Args&&...args):
    data(d), dim{{std::forward<Args>(args)...}}
  {}
};

template <typename DATA, size_t DIM=2>
struct KDTreeNodes {
  std::vector<float> median; // or dimCurrent;
  std::vector<int> right;
  std::vector<std::array<float, DIM-1> > dimOthers;
  std::vector<DATA> data;

  int poolSize;
  int poolPos;

  constexpr KDTreeNodes(): poolSize(-1), poolPos(-1) {}

  bool empty() const { return poolPos == -1; }
  int size() const { return poolPos + 1; }

  void clear() {
    median.clear();
    right.clear();
    dimOthers.clear();
    data.clear();
    poolSize = -1;
    poolPos = -1;
  }

  int getNextNode() {
    ++poolPos;
    return poolPos;
  }

  void build(int sizeData) {
    poolSize = sizeData*2-1;
    median.resize(poolSize);
    right.resize(poolSize);
    dimOthers.resize(poolSize);
    data.resize(poolSize);
  };

  constexpr bool isLeaf(int right) const {
    // Valid values of right are always >= 2
    // index 0 is the root, and 1 is the first left node
    // Exploit index values 0 and 1 to mark which of dim1/dim2 is the
    // current one in recSearch() at the depth of the leaf.
    return right < 2;
  }

  bool isLeafIndex(int index) const {
    return isLeaf(right[index]);
  }
};

namespace kdtreetraits {
  template <size_t INDEX, size_t DIM>
  bool isInside_(std::array<float, DIM-1> dimOthers,
                 std::array<std::tuple<float, float>, DIM-1> dimOthersLimits) {
    return (dimOthers[INDEX] >= std::get<0>(dimOthersLimits[INDEX])) &
           (dimOthers[INDEX] <= std::get<1>(dimOthersLimits[INDEX]));
  }
  template <size_t INDEX, size_t DIM>
  struct IsInside_ {
    static bool call(std::array<float, DIM-1> dimOthers,
                     std::array<std::tuple<float, float>, DIM-1> dimOthersLimits) {
      return IsInside_<INDEX-1, DIM>::call(dimOthers, dimOthersLimits) &
        isInside_<INDEX, DIM>(dimOthers, dimOthersLimits);
    }
  };
  // break recursion
  template <size_t DIM>
  struct IsInside_<0, DIM> {
    static bool call(std::array<float, DIM-1> dimOthers,
                     std::array<std::tuple<float, float>, DIM-1> dimOthersLimits) {
      return isInside_<0, DIM>(dimOthers, dimOthersLimits);
    }
  };
}
template <typename DATA, size_t DIM>
struct KDTreeTraits {
  static void rewindIndices(const std::vector<KDTreeNodeInfo<DATA, DIM> >& initialList, const KDTreeNodeInfo<DATA, DIM>& item, int& i, int& j, const int dimIndex) {
    while (initialList[i].dim[dimIndex] < item.dim[dimIndex]) i++;
    while (initialList[j].dim[dimIndex] > item.dim[dimIndex]) j--;
  }

  static bool isInside(std::array<float, DIM-1> dimOthers,
                       std::array<std::tuple<float, float>, DIM-1> dimOthersLimits) {
    return kdtreetraits::IsInside_<DIM-2, DIM>::call(dimOthers, dimOthersLimits);
  }

  static void swapLimits(int depth, std::tuple<float, float>& dimCurrLimits,
                         std::array<std::tuple<float, float>, DIM-1>& dimOthersLimits) {
    const int dimIndex = depth % (DIM-1);
    std::swap(dimCurrLimits, dimOthersLimits[dimIndex]);
  }
};
// Specialization for 2D for performance
template <typename DATA>
struct KDTreeTraits<DATA, 2> {
  inline
  static void rewindIndices(const std::vector<KDTreeNodeInfo<DATA, 2> >& initialList, const KDTreeNodeInfo<DATA, 2>& item, int& i, int& j, const int dimIndex) {
    if(dimIndex == 0) {
      while (initialList[i].dim[0] < item.dim[0]) i++;
      while (initialList[j].dim[0] > item.dim[0]) j--;
    }
    else {
      while (initialList[i].dim[1] < item.dim[1]) i++;
      while (initialList[j].dim[1] > item.dim[1]) j--;
    }
  }

  inline
  static bool isInside(std::array<float, 1> dimOthers,
                       std::array<std::tuple<float, float>, 1> dimOthersLimits) {
    return kdtreetraits::IsInside_<0, 2>::call(dimOthers, dimOthersLimits);
  }

  inline
  static void swapLimits(int depth, std::tuple<float, float>& dimCurrLimits,
                         std::array<std::tuple<float, float>, 1>& dimOthersLimits) {
    std::swap(dimCurrLimits, dimOthersLimits[0]);
  }
};
#endif

#include "RecoPixelVertexing/PixelTriplets/plugins/KDTree.h"

#include <iostream>
#include <time.h>

namespace {
  double delta(const struct timespec & first, const struct timespec & second)
  {
    if (second.tv_nsec > first.tv_nsec)
      return (double) (second.tv_sec - first.tv_sec) + (double) (second.tv_nsec - first.tv_nsec) / (double) 1e9;
    else
      return (double) (second.tv_sec - first.tv_sec) - (double) (first.tv_nsec - second.tv_nsec) / (double) 1e9;
  }


  void runTest() {
    constexpr int REPEAT=400;
    constexpr int N=10000;

    struct timespec start;
    struct timespec stop;
    double build_time = 0.0;
    double search_time = 0.0;

    for(int iRep=0; iRep<REPEAT; ++iRep) {
      // create data points
      std::vector<KDTreeNodeInfo<int> > nodes;
      nodes.reserve(N);
      int id = 0;
      for(int i=0; i<100; ++i) {
        for(int j=0; j<100; ++j) {
          nodes.emplace_back(id, i*0.1f, j*0.1f);
        }
      }

      // build tree
      KDTree<int> kdtree;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
      kdtree.build(nodes);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
      build_time += delta(start, stop);

      // search
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
      std::vector<int> values;
      for(const auto& node: nodes) {
        kdtree.search(KDTreeBox<>(node.dim[0]-0.1f, node.dim[0]+0.1f, node.dim[1]-0.1f, node.dim[1]+0.1f), values);
        values.clear();
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
      search_time += delta(start, stop);
    }

    std::cout << "Total time in KDTree<>::build() " << build_time << " s" << std::endl;
    std::cout << "Total time in KDTree<>::search() " << search_time << " s" << std::endl;
  }
}

int main() {
  runTest();
  return 0;
}

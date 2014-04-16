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

  template <size_t DIM>
  struct testTraits {};

  template <> struct testTraits<2> {
    const static size_t items_per_dim = 100;
    const static size_t n_items = 10000;
    const static size_t repeat = 400;
  };
  template <> struct testTraits<3> {
    const static size_t items_per_dim = 20;
    const static size_t n_items = 8000;
    const static size_t repeat = 200;
  };
  template <> struct testTraits<4> {
    const static size_t items_per_dim = 10;
    const static size_t n_items = 10000;
    const static size_t repeat = 50;
  };
  template <> struct testTraits<5> {
    const static size_t items_per_dim = 6;
    const static size_t n_items = 7776;
    const static size_t repeat = 20;
  };
  template <> struct testTraits<6> {
    const static size_t items_per_dim = 5;
    const static size_t n_items = 15625;
    const static size_t repeat = 5;
  };
  template <> struct testTraits<7> {
    const static size_t items_per_dim = 4;
    const static size_t n_items = 16384;
    const static size_t repeat = 2;
  };
  template <> struct testTraits<8> {
    const static size_t items_per_dim = 3;
    const static size_t n_items = 6561;
    const static size_t repeat = 2;
  };

  template <size_t DIM>
  void runTest() {
    struct timespec start;
    struct timespec stop;
    double build_time = 0.0;
    double search_time = 0.0;

    for(size_t iRep=0; iRep<testTraits<DIM>::repeat; ++iRep) {
      // create data points
      std::vector<KDTreeNodeInfo<int, DIM> > nodes;
      nodes.reserve(testTraits<DIM>::n_items);
      std::array<size_t, DIM> coordinates{};
      for(size_t id=0; id<testTraits<DIM>::n_items; ++id) {
        std::array<float, DIM> coords;
        for(size_t i=0; i<DIM; ++i) {
          coords[i] = coordinates[i]*0.1f;
        }

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

      // build tree
      KDTree<int, DIM> kdtree;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
      kdtree.build(nodes);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
      build_time += delta(start, stop);

      // search
      std::array<std::tuple<float, float>, DIM> box;
      std::vector<int> values;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
      for(const auto& node: nodes) {
        for(size_t i=0; i<DIM; ++i) {
          box[i] = std::make_tuple(node.dim[i]-0.1f, node.dim[i]+0.1f);
        }
        kdtree.search(KDTreeBox<DIM>(box), values);
        values.clear();
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
      search_time += delta(start, stop);
    }

    //std::cout << "Total time in KDTree<DIM=" << DIM << ">::build() " << build_time << " s" << std::endl;
    //std::cout << "Total time in KDTree<DIM=" << DIM << ">::search() " << search_time << " s" << std::endl;
    std::cout << DIM << " (" << testTraits<DIM>::repeat << " reps)   " << build_time << "                   " << search_time << std::endl;
  }
}

int main() {
  std::cout << "Dimension    KDTree<DIM>::build() (s)   KDTree<DIM>::search() (s)" << std::endl;
  runTest<2>();
  runTest<3>();
  runTest<4>();
  runTest<5>();
  runTest<6>();
  runTest<7>();
  runTest<8>();
  return 0;
}

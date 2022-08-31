#ifndef DataFormats_SiStripCluster_SiStripApproximateClusterCollection_h
#define DataFormats_SiStripCluster_SiStripApproximateClusterCollection_h

#include <iterator>
#include <vector>

#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"

/**
 * This class provides a minimal interface that resembles
 * edmNew::DetSetVector, but is crafted such that we are comfortable
 * to provide an infinite backwards compatibility guarantee for it
 * (like all RAW data). Any modifications need to be made with care.
 * Please consult core software group if in doubt.
 */
class SiStripApproximateClusterCollection {
public:
  // Helper classes to make creation and iteration easier
  class Filler {
  public:
    void push_back(SiStripApproximateCluster const& cluster) {
      clusters_.push_back(cluster);
      ++offsetToEnd_;
    }

  private:
    friend SiStripApproximateClusterCollection;
    Filler(std::vector<SiStripApproximateCluster>& clusters, unsigned short& offsetToEnd)
        : clusters_(clusters), offsetToEnd_(offsetToEnd) {}

    std::vector<SiStripApproximateCluster>& clusters_;
    unsigned short& offsetToEnd_;
  };

  class const_iterator;
  class DetSet {
  public:
    using const_iterator = std::vector<SiStripApproximateCluster>::const_iterator;

    unsigned int id() const { return id_; }

    const_iterator begin() const { return clusBegin_; }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return clusEnd_; }
    const_iterator cend() const { return end(); }

  private:
    friend SiStripApproximateClusterCollection::const_iterator;
    DetSet(unsigned int id, const_iterator begin, const_iterator end) : id_(id), clusBegin_(begin), clusEnd_(end) {}

    unsigned int id_;
    const_iterator clusBegin_;
    const_iterator clusEnd_;
  };

  class const_iterator {
  public:
    DetSet operator*() const { return DetSet(coll_->detIds_[detIndex_], clusBegin_, clusEnd_); }

    const_iterator& operator++() {
      ++detIndex_;
      if (detIndex_ == coll_->detIds_.size()) {
        *this = const_iterator();
      } else {
        clusBegin_ = clusEnd_;
        clusEnd_ = std::next(clusBegin_, coll_->offsetsToEnd_[detIndex_]);
      }
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator clone = *this;
      ++(*this);
      return clone;
    }

    bool operator==(const_iterator const& other) const { return coll_ == other.coll_ and detIndex_ == other.detIndex_; }
    bool operator!=(const_iterator const& other) const { return not operator==(other); }

  private:
    friend SiStripApproximateClusterCollection;
    // default-constructed object acts as the sentinel
    const_iterator() = default;
    const_iterator(SiStripApproximateClusterCollection const* coll) : coll_(coll) {}

    using clusterIterator = std::vector<SiStripApproximateCluster>::const_iterator;

    SiStripApproximateClusterCollection const* coll_ = nullptr;
    unsigned int detIndex_ = 0;
    clusterIterator clusBegin_;
    clusterIterator clusEnd_;
  };

  // Actual public interface

  SiStripApproximateClusterCollection() = default;

  void reserve(size_t dets, size_t clusters);
  Filler beginDet(unsigned int detId);

  const_iterator begin() const { return const_iterator(this); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(); }
  const_iterator cend() const { return end(); }

private:
  // DetID for the Det
  std::vector<unsigned int> detIds_;  // DetId for the Det

  // Offset to the *end* of the Det
  // unsigned short is fine as a module can not have more than 65536
  // clusters. But see note below for possible further evolution
  std::vector<unsigned short> offsetsToEnd_;

  // Note: if you would be sure that there would be no more than 2^7 =
  // 128 clusters on a given SiStrip module, you could use the highest
  // 7 bits of the DetID to store the offset (the Det and SubDet parts
  // of the DetId practically hardcoded here). Looking at the SiStrip
  // tracker DetId definition there is actually even more free space,
  // 12 bits, which would give you space for 2^12 = 4096 clusters per
  // module (which is already more than possible).
  std::vector<SiStripApproximateCluster> clusters_;
};

#endif

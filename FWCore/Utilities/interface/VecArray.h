#ifndef FWCore_Utilities_VecArray_h
#define FWCore_Utilities_VecArray_h

#include <array>
#include <utility>
#include <stdexcept>
#include <string>

namespace edm {
  /**
   * A class for extending std::array with std::vector-like interface.
   *
   * This class can be useful if the maximum length is known at
   * compile-time (can use std::array), and that the length is rather
   * small (maximum size of std::array is comparable with the overhead
   * of std::vector). It is also free of dynamic memory allocations.
   *
   * Note that the implemented interface is not complete compared to
   * std::array or std:vector. Feel free contribute if further needs
   * arise.
   *
   * The second template argument is unsigned int (not size_t) on
   * purpose to reduce the size of the class by 4 bytes (at least in
   * Linux amd64). For all practical purposes even unsigned char could
   * be enough.
   */
  template <typename T, unsigned int N>
  class VecArray {
    using Data = std::array<T, N>;
    Data data_;
    unsigned int size_;

  public:
    using value_type = typename Data::value_type;
    using size_type = unsigned int;
    using difference_type = typename Data::difference_type;
    using reference = typename Data::reference;
    using const_reference = typename Data::const_reference;
    using pointer = typename Data::pointer;
    using const_pointer = typename Data::const_pointer;
    using iterator = typename Data::iterator;
    using const_iterator = typename Data::const_iterator;
    //using typename Data::reverse_iterator;
    //using typename Data::const_reverse_iterator;

    VecArray(): data_{}, size_{0} {}

    // Not range-checked, undefined behaviour if access beyond size()
    reference operator[](size_type pos) { return data_[pos]; }
    // Not range-checked, undefined behaviour if access beyond size()
    const_reference operator[](size_type pos) const { return data_[pos]; }
    // Undefined behaviour if size()==0
    reference front() { return data_[0]; }
    // Undefined behaviour if size()==0
    const_reference front() const { return data_[0]; }

    // Undefined behaviour if size()==0
    reference back() { return operator[](size_-1); }
    // Undefined behaviour if size()==0
    const_reference back() const { return operator[](size_-1); }
    pointer data() { return data_.data(); }
    const_pointer data() const { return data_.data(); }

    iterator begin() noexcept { return data_.begin(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }

    iterator end() noexcept { return begin()+size_; }
    const_iterator end() const noexcept { return begin()+size_; }
    const_iterator cend() const noexcept { return cbegin()+size_; }

    constexpr bool empty() noexcept { return size_ == 0; }
    constexpr size_type size() noexcept { return size_; }
    constexpr size_type maxSize() noexcept { return N; }

    void clear() {
      size_ = 0;
    }

    // Undefined behaviour if size()==N
    void push_back(const T& value) {
      data_[size_] = value;
      ++size_;
    }

    // Undefined behaviour if size()==0
    void pop_back() {
      --size_;
    }

    void resize(unsigned int size) {
      if(size > N)
        throw std::length_error("Requesting size "+std::to_string(size)+" while maximum allowed is "+std::to_string(N));

      while(size < size_)
        pop_back();
      size_ = size;
    }

    void swap(VecArray& other)
      noexcept(noexcept(data_.swap(other.data_)) &&
               noexcept(std::swap(size_, other.size_)))
    {
      data_.swap(other.data_);
      std::swap(size_, other.size_);
    }
  };
}

#endif

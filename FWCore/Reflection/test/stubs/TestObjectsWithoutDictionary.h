#ifndef FWCore_Reflection_test_TestObjectsWithoutDictionary_h
#define FWCore_Reflection_test_TestObjectsWithoutDictionary_h

namespace edmtest::reflection {
  class IntObjectWithoutDictionary {
  public:
    IntObjectWithoutDictionary();
    IntObjectWithoutDictionary(int v) : value_(v) {}

    int get() const { return value_; }

  private:
    int value_ = 0;
  };
}

#endif

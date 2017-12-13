#ifndef FWCore_Utilities_TransientEDGetToken_h
#define FWCore_Utilities_TransientEDGetToken_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     TransientEDGetToken
// 
/**\class TransientEDGetToken TransientEDGetToken.h "FWCore/Utilities/interface/TransientEDGetToken.h"

 Description: A Token used to get data from the EDM

 Usage:
    A TransientEDGetToken is created by calls to 'consumes' or 'mayConsume' from an EDM module.
 The TransientEDGetToken can then be used to quickly retrieve data from the edm::Event, edm::LuminosityBlock or edm::Run.
 
The templated form, TransientEDGetTokenT<T>, is the same as EDGetToken except when used to get data the framework
 will skip checking that the type being requested matches the type specified during the 'consumes' or 'mayConsume' call.

Differs from EDGetToken do denote transient reads from the EDM (i.e.
access the product only during the filter/produce/analyze)

*/

// system include files

// user include files

// forward declarations
namespace edm {
  class EDConsumerBase;
  template <typename T> class TransientEDGetTokenT;
  
  class TransientEDGetToken
  {
    friend class EDConsumerBase;
    
  public:
    
    TransientEDGetToken() : m_value{s_uninitializedValue} {}

    template<typename T>
    TransientEDGetToken(TransientEDGetTokenT<T> iOther): m_value{iOther.m_value} {}

    // ---------- const member functions ---------------------
    unsigned int index() const { return m_value; }
    bool isUninitialized() const { return m_value == s_uninitializedValue; }

  private:
    //for testing
    friend class TestEDGetToken;
    
    static const unsigned int s_uninitializedValue = 0xFFFFFFFF;

    explicit TransientEDGetToken(unsigned int iValue) : m_value(iValue) { }

    // ---------- member data --------------------------------
    unsigned int m_value;
  };

  template<typename T>
  class TransientEDGetTokenT
  {
    friend class EDConsumerBase;
    friend class TransientEDGetToken;

  public:

    TransientEDGetTokenT() : m_value{s_uninitializedValue} {}
  
    // ---------- const member functions ---------------------
    unsigned int index() const { return m_value; }
    bool isUninitialized() const { return m_value == s_uninitializedValue; }

  private:
    //for testing
    friend class TestEDGetToken;

    static const unsigned int s_uninitializedValue = 0xFFFFFFFF;

    explicit TransientEDGetTokenT(unsigned int iValue) : m_value(iValue) { }

    // ---------- member data --------------------------------
    unsigned int m_value;
  };
}

#endif

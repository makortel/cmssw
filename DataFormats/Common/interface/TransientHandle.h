#ifndef DataFormats_Common_TransientHandle_h
#define DataFormats_Common_TransientHandle_h

/*----------------------------------------------------------------------
  
Handle: Non-owning "smart pointer" for reference to Products and
their Provenances.

This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to Product or Provenance is destroyed, use of the
Handle becomes undefined. There is no way to query the Handle to
discover if this has happened.

Handles can have:
  -- Product and Provenance pointers both null;
  -- Both pointers valid

To check validity, one can use the isValid() function.

If failedToGet() returns true then the requested data is not available
If failedToGet() returns false but isValid() is also false then no attempt 
  to get data has occurred

Denotes a read from Event where the product is accessed only during
the filter/producer/analyze (i.e.not stored in any data products or
member variables with longer life time). Functionally equivalent to
Handle, but is defined as a separate type to prevent creation of
Ref/RefToBase/Ptr from it.

----------------------------------------------------------------------*/
#include <typeinfo>

#include "DataFormats/Common/interface/HandleBase.h"

namespace edm {

  template <typename T>
  class TransientHandle : public HandleBase {
  public:
    typedef T element_type;

    // Default constructed handles are invalid.
    TransientHandle();

    TransientHandle(T const* prod, Provenance const* prov);
    
    TransientHandle(std::shared_ptr<HandleExceptionFactory> &&);
    TransientHandle(TransientHandle const&) = default;
    
    TransientHandle& operator=(TransientHandle&&) = default;
    TransientHandle& operator=(TransientHandle const&) = default;
    
    ~TransientHandle() = default;

    T const* product() const;
    T const* operator->() const; // alias for product()
    T const& operator*() const;

  private:
  };

  template <class T>
  TransientHandle<T>::TransientHandle() : HandleBase()
  { }

  template <class T>
  TransientHandle<T>::TransientHandle(T const* prod, Provenance const* prov) : HandleBase(prod, prov) { 
  }

  template <class T>
  TransientHandle<T>::TransientHandle(std::shared_ptr<edm::HandleExceptionFactory> && iWhyFailed) :
  HandleBase(std::move(iWhyFailed))
  { }

  template <class T>
  T const* 
  TransientHandle<T>::product() const { 
    return static_cast<T const*>(productStorage());
  }

  template <class T>
  T const* 
  TransientHandle<T>::operator->() const {
    return product();
  }

  template <class T>
  T const& 
  TransientHandle<T>::operator*() const {
    return *product();
  }
}
#endif

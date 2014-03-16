#ifndef FWCore_PluginManager_PluginFactory_h
#define FWCore_PluginManager_PluginFactory_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginFactory
// 
/**\class PluginFactory PluginFactory.h FWCore/PluginManager/interface/PluginFactory.h

 Description: Public interface for loading a plugin

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Apr  5 12:10:23 EDT 2007
//

// system include files
#include <functional>
#include <map>
#include <vector>

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
// forward declarations
namespace edm {
  class ConfigurationDescriptions;
}

namespace edmplugin {
template< class T> class PluginFactory;
  class DummyFriend;
  
  namespace impl {
    // http://stackoverflow.com/questions/9530928/checking-a-member-exists-possibly-in-a-base-class-c11-versio
    // http://stackoverflow.com/questions/257288/is-it-possible-to-write-a-c-template-to-check-for-a-functions-existence/264088
    template <typename TPlug, int>
    constexpr auto has_fillDescriptions(edm::ConfigurationDescriptions& descriptions) -> decltype(TPlug::fillDescriptions(descriptions), bool()) {
      return true;
    }
    template <typename TPlug, long>
    constexpr bool has_fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      return false;
    }
    template <typename TPlug, bool>
    struct CallFillDescriptions {
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions, std::function<void(edm::ConfigurationDescriptions&)> defaultFunction) {
        TPlug::fillDescriptions(descriptions);
      }
    };
    template <typename TPlug>
    struct CallFillDescriptions<TPlug, false> {
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions, std::function<void(edm::ConfigurationDescriptions&)> defaultFunction) {
        defaultFunction(descriptions);
      }
    };
  }

template<typename R, typename... Args>
class PluginFactory<R*(Args...)> : public PluginFactoryBase
{
      friend class DummyFriend;
   public:
      typedef R* TemplateArgType(Args...);
      typedef R* ReturnType;

      struct PMakerBase {
        typedef std::function<void(edm::ConfigurationDescriptions&)> CallbackType;
        virtual R* create(Args...) const = 0;
        virtual void fillDescriptions(edm::ConfigurationDescriptions& descriptions, CallbackType defaultFunction, CallbackType prevalidateFunction) const = 0;
        virtual ~PMakerBase() {}
      };
      template<class TPlug>
      struct PMaker : public PMakerBase {
        typedef typename PMakerBase::CallbackType CallbackType;
        PMaker(const std::string& iName) {
          PluginFactory<R*(Args...)>::get()->registerPMaker(this,iName);
        }
        virtual R* create(Args... args) const {
          return new TPlug(std::forward<Args>(args)...);
        }
        // dirty solution with callbacks in order to avoid circular
        // dependence between PluginManager and ParameterSet packages
        virtual void fillDescriptions(edm::ConfigurationDescriptions& descriptions, CallbackType defaultFunction, CallbackType prevalidateFunction) const override {
          impl::CallFillDescriptions<TPlug, impl::has_fillDescriptions<TPlug, 0>(descriptions)>::fillDescriptions(descriptions, defaultFunction);
          prevalidateFunction(descriptions);
        }
      };

      // ---------- const member functions ---------------------
      virtual const std::string& category() const ;
      
      R* create(const std::string& iName, Args... args) const {
        return reinterpret_cast<PMakerBase*>(PluginFactoryBase::findPMaker(iName))->create(std::forward<Args>(args)...);
      }

      ///like above but returns 0 if iName is unknown
      R* tryToCreate(const std::string& iName, Args... args) const {
        auto found = PluginFactoryBase::tryToFindPMaker(iName);
        if(found ==nullptr) {
          return nullptr;
        }
        return reinterpret_cast<PMakerBase*>(found)->create(args...);
      }
      // ---------- static member functions --------------------

      static PluginFactory<R*(Args...)>* get();
      // ---------- member functions ---------------------------
      void registerPMaker(PMakerBase* iPMaker, const std::string& iName) {
        PluginFactoryBase::registerPMaker(iPMaker, iName);
      }

   private:
      PluginFactory() {
        finishedConstruction();
      }
      PluginFactory(const PluginFactory&) = delete; // stop default

      const PluginFactory& operator=(const PluginFactory&) = delete; // stop default

};
}
#define CONCATENATE_HIDDEN(a,b) a ## b 
#define CONCATENATE(a,b) CONCATENATE_HIDDEN(a,b)
#define EDM_REGISTER_PLUGINFACTORY(_factory_,_category_) \
namespace edmplugin {\
  template<> edmplugin::PluginFactory<_factory_::TemplateArgType>* edmplugin::PluginFactory<_factory_::TemplateArgType>::get() { [[cms::thread_safe]] static edmplugin::PluginFactory<_factory_::TemplateArgType> s_instance; return &s_instance;}\
  template<> const std::string& edmplugin::PluginFactory<_factory_::TemplateArgType>::category() const { static const std::string s_cat(_category_);  return s_cat;}\
  } enum {CONCATENATE(dummy_edm_register_pluginfactory_, __LINE__)}

#endif

#define EDM_PLUGIN_SYM(x,y) EDM_PLUGIN_SYM2(x,y)
#define EDM_PLUGIN_SYM2(x,y) x ## y

#define DEFINE_EDM_PLUGIN(factory,type,name) \
static const factory::PMaker<type> EDM_PLUGIN_SYM(s_maker , __LINE__ ) (name)


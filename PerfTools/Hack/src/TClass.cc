//#include "TClass.h"
//#include "TCollection.h"
//#include "TListOfDataMembers.h"
//#include "TDataMember.h"

#include <iostream>
#include <cassert>

#include <cstring>

#include <dlfcn.h>

class TClass {
public:
  static TClass* GetClass(char const*, bool, bool, unsigned long, unsigned long);
};

TClass *TClass::GetClass(char const* a, bool b, bool c, unsigned long d, unsigned long e) {
  void* original = dlsym(RTLD_NEXT, "_ZN6TClass8GetClassEPKcbbmm");
  assert(original);
  auto* casted = reinterpret_cast<TClass*(*)(char const*, bool, bool, unsigned long, unsigned long)>(original);

  void* orig_hackprint = dlsym(RTLD_DEFAULT, "g_TClassHackPrint");
  int* g_TClassHackPrint = nullptr;
  if (orig_hackprint) {
    g_TClassHackPrint = reinterpret_cast<int*>(orig_hackprint);
  }

  if (g_TClassHackPrint and *g_TClassHackPrint > 0) {
    std::cerr << "hack begin: " << a << std::endl;
    assert(strcmp(a, "TObjArray") != 0);
  }
  auto ret = casted(a, b, c, d, e);
  if (g_TClassHackPrint and *g_TClassHackPrint > 0) {
    std::cerr << "hack end: " << a << std::endl;
  }
  return ret;
}

/*
class TCling {
  int AutoParse(const char* cls);
};

int TCling::AutoParse(char const*) {
  
}
*/

#ifdef FOO
void TClass::GetMissingDictionariesForMembers(TCollection& result, TCollection& visited, bool recurse)
{
  {
  }
   TListOfDataMembers* ldm = (TListOfDataMembers*)GetListOfDataMembers();
   if (!ldm) return ;
   TIter nextMemb(ldm);
   TDataMember * dm = nullptr;
   while ((dm = (TDataMember*)nextMemb())) {
      // If it is a transient
      if(!dm->IsPersistent()) {
        continue;
      }
      if (dm->Property() & kIsStatic) {
         continue;
      }
      // If it is a built-in data type.
      TClass* dmTClass = nullptr;
      if (dm->GetDataType()) {
         // We have a basic datatype.
         dmTClass = nullptr;
         // Otherwise get the string representing the type.
      } else if (dm->GetTypeName()) {
        std::cout << "hack: " << dm->GetTypeName() << std::endl;
         dmTClass = TClass::GetClass(dm->GetTypeName());
      }
      if (dmTClass) {
         dmTClass->GetMissingDictionariesWithRecursionCheck(result, visited, recurse);
      }
   }
}
#endif

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "TBaseClass.h"
#include "TClass.h"
#include "TError.h"
#include "TEnv.h"
#include "THashTable.h"
#include "TInterpreter.h"
#include "TList.h"

auto originalErrorHandler() {
  static auto handler = GetErrorHandler();
  return handler;
}

void RootErrorHandler(int level, bool b, char const* location, char const* message) {
  if (level >= kWarning) {
    throw std::runtime_error(std::string(message) + " (from " + location + ")");
  }
  originalErrorHandler()(level, b, location, message);
}
bool hasHelp(const char* arg) {
  std::string_view view(arg);
  return view == "-h" or view == "--help";
}

/*
bool verifyBaseClasses(TClass* cl) {
  if (not cl) {
    return true;
  }
  TList* bases = cl->GetListOfBases();
  if (not bases) {
    return true;
  }
  bool success = true;
  for (auto* base : *bases) {
    TBaseClass* baseClass = dynamic_cast<TBaseClass*>(base);
    if (not baseClass) {
      std::cout << "Invalid base class for " << cl->GetName() << std::endl;
      success = false;
      continue;
    }
    auto* baseCl = baseClass->GetClassPointer();
    if (baseCl) {
      if (not verifyBaseClasses(baseCl)) {
        success = false;
      }
    } else {
      std::cout << "Incomplete base class for " << cl->GetName() << ": " << baseClass->GetName() << std::endl;
      success = false;
    }
  }
  return success;
}
  */

int main(int argc, char** argv) {
  if (argc == 1 or (argc == 2 and hasHelp(argv[1]))) {
    std::cout << "Usage: edmCheckWrappers <list of edm::Wrapper class names>\n"
              << "Checks that TClass::GetClass() works for each argument class name without header autoparsing"
              << std::endl;
    return EXIT_SUCCESS;
  }

  bool success = true;

  originalErrorHandler();
  //SetErrorHandler(RootErrorHandler);
  //gInterpreter->SetClassAutoloading(true);
  gInterpreter->SetClassAutoparsing(false);
  //gEnv->SetValue("Root.TClass.GetClass.AutoParsing", true);

  for (int i = 1; i < argc; ++i) {
    TClass* cl = nullptr;
    try {
      cl = TClass::GetClass(argv[i]);
      if (cl) {
        std::cout << "Found TClass for " << cl->GetName() << std::endl;
        THashTable hashTable;
        bool recursive = true;
        cl->GetMissingDictionaries(hashTable, recursive);
        for (auto const& item : hashTable) {
          std::cout << "Missing dictionary for " << item->GetName() << std::endl;
          success = false;
        }

        /*
        [[maybe_unused]] auto const* streamer = cl->GetStreamerInfo();
        if (not verifyBaseClasses(cl)) {
          success = false;
        }
          */
      }
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
      success = false;
      continue;
    }
    if (not cl) {
      std::cout << "No TClass for " << argv[i] << std::endl;
      success = false;
    }
  }

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

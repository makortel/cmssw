#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "TBaseClass.h"
#include "TBufferFile.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TError.h"
#include "TEnv.h"
#include "THashTable.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TMemFile.h"
#include "TTree.h"

// Would be good to avoid...
#include "DataFormats/Common/interface/WrapperBase.h"
#include "FWCore/Utilities/interface/getAnyPtr.h"

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

// Round-trips obj through a ROOT streamer to a memory buffer and back, verifying the streamer works.
bool checkStreamerRoundTrip(TClass* cl, void* obj) {
  TBufferFile writeBuffer(TBuffer::kWrite);
  cl->Streamer(obj, writeBuffer);

  TBufferFile readBuffer(TBuffer::kRead, writeBuffer.BufferSize(), writeBuffer.Buffer(), kFALSE);
  void* newObj = cl->New();
  if (not newObj) {
    std::cout << "Could not construct object for streamer read-back of " << cl->GetName() << std::endl;
    return false;
  }
  cl->Streamer(newObj, readBuffer);
  cl->Destructor(newObj);
  return true;
}

// Round-trips obj through a TTree branch stored in a ROOT in-memory file, verifying (de)serialization works.
bool checkTTreeRoundTrip(TClass* cl, void* obj) {
  TMemFile file("checkTTreeRoundTrip.root", "RECREATE");
  TTree tree("t", "t");
  int splitlevel = 0;
  TBranch* branch = tree.Branch(cl->GetName(), cl->GetName(), &obj, 32000, splitlevel);
  if (not branch) {
    std::cout << "Could not create TTree branch for " << cl->GetName() << std::endl;
    return false;
  }
  tree.Fill();

  void* newObj = cl->New();
  if (not newObj) {
    std::cout << "Could not construct object for TTree read-back of " << cl->GetName() << std::endl;
    return false;
  }
  // Pass cl explicitly: the void* template deduction can't identify the real class.
  tree.SetBranchAddress(cl->GetName(), &newObj, cl, kOther_t, true);
  tree.GetEntry(0);
  cl->Destructor(newObj);
  return true;
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

bool checkMissingDictionaries(TClass *cl) {
  if (not cl) {
    std::cout << "TClass is null" << std::endl;
    return false;
  }
  bool success = true;

  THashTable hashTable;
  bool recursive = true;
  cl->GetMissingDictionaries(hashTable, recursive);
  for (auto const& item : hashTable) {
    std::cout << "Missing dictionary for " << item->GetName() << std::endl;
    success = false;
  }

  std::vector<std::string> args;
  int nestedLoc = 0;
  TClassEdit::GetSplit(cl->GetName(), args, nestedLoc);
  if (not args.empty()) {
    if (args.front() == "edm::Wrapper") {
      std::cout << "Recursing to template argument " << args[1] << std::endl;
      success = success & checkMissingDictionaries(TClass::GetClass(args[1].c_str()));
    } else if (args.front() == "vector") {
      std::cout << "Recursing to template argument " << args[1] << std::endl;
      success = success & checkMissingDictionaries(TClass::GetClass(args[1].c_str()));
    } else if (args.front() == "unordered_map") {
      std::cout << "Recursing to template argument " << args[1] << std::endl;
      success = success & checkMissingDictionaries(TClass::GetClass(args[1].c_str()));
      std::cout << "Recursing to template argument " << args[2] << std::endl;
      success = success & checkMissingDictionaries(TClass::GetClass(args[2].c_str()));
    } else if (args.front() == "pair") {
      std::cout << "Recursing to template argument " << args[1] << std::endl;
      success = success & checkMissingDictionaries(TClass::GetClass(args[1].c_str()));
      std::cout << "Recursing to template argument " << args[2] << std::endl;
      success = success & checkMissingDictionaries(TClass::GetClass(args[2].c_str()));
    }
  }
  
  return success;
}

int main(int argc, char** argv) {
  if (argc == 1 or (argc == 2 and hasHelp(argv[1]))) {
    std::cout << "Usage: edmCheckWrappers <list of edm::Wrapper class names>\n"
              << "Checks that TClass::GetClass() works for each argument class name without header autoparsing"
              << std::endl;
    return EXIT_SUCCESS;
  }

  bool success = true;
  constexpr bool autoParsing = true;
  //constexpr bool autoParsing = false;

  originalErrorHandler();
  SetErrorHandler(RootErrorHandler);
  //gInterpreter->SetClassAutoloading(true);
  gInterpreter->SetClassAutoparsing(autoParsing);
  //gEnv->SetValue("Root.TClass.GetClass.AutoParsing", true);

  std::cout << "Auto-parsing is enabled? " << autoParsing << std::endl;

  auto wrapperBase = TClass::GetClass("edm::WrapperBase");
  if (not wrapperBase) {
    std::cout << "Could not find TClass for edm::WrapperBase, something is badly wrong!" << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 1; i < argc; ++i) {
    /*
    std::string_view className(argv[i]);
    if (className.starts_with("edm::Wrapper<") and className.ends_with(">")) {
      std::cout << "Checking " << className << std::endl;
    } else {
      std::cout << "Skipping " << className << " (not an edm::Wrapper)" << std::endl;
      continue;
    }
      */
    TClass* cl = nullptr;
    try {
      cl = TClass::GetClass(argv[i]);
      if (cl) {
        std::cout << "Found TClass for " << cl->GetName() << std::endl;
        /*
        THashTable hashTable;
        bool recursive = true;
        cl->GetMissingDictionaries(hashTable, recursive);
        for (auto const& item : hashTable) {
          std::cout << "Missing dictionary for " << item->GetName() << std::endl;
          success = false;
          }*/
        success = success & checkMissingDictionaries(cl);

        void* obj = cl->New();
        if (not obj) {
          std::cout << " construction failed" << std::endl;
          success = false;
          continue;
        }
        /*
        int offset = cl->GetBaseClassOffset(wrapperBase);
        std::unique_ptr<edm::WrapperBase> dummy = edm::getAnyPtr<edm::WrapperBase>(obj, offset);
        */

        if (not checkStreamerRoundTrip(cl, obj)) {
          success = false;
        }

        if (not checkTTreeRoundTrip(cl, obj)) {
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

  std::cout << "Auto-parsing report" << std::endl;
  gInterpreter->Print("autoparsed");

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

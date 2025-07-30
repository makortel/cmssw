#include <cstdlib>
#include <iostream>
#include <string>

#include "TClass.h"
#include "TError.h"
#include "TInterpreter.h"

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


int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << "Usage: edmCheckWrappers <list of edm::Wrapper class names>\n"
              << "Checks that TClass::GetClass() works for each argument class name without header autoparsing" << std::endl;
    return EXIT_SUCCESS;
  }

  int success = EXIT_SUCCESS;
  
  originalErrorHandler();
  SetErrorHandler(RootErrorHandler);
  //gInterpreter->SetClassAutoloading(true);
  gInterpreter->SetClassAutoparsing(false);

  for (int i=1; i<argc; ++i) {
    TClass const* cl = nullptr;
    try {
      cl = TClass::GetClass(argv[i]);
    } catch(std::runtime_error& e) {
      std::cout << e.what() << std::endl;
      success = EXIT_FAILURE;
      continue;
    }
    if (not cl) {
      std::cout << "No TClass for " << argv[i] << std::endl;
      success = EXIT_FAILURE;
    }
  }
  
  return success;
}

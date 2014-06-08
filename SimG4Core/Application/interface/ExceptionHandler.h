#ifndef SimG4Core_ExceptionHandler_H
#define SimG4Core_ExceptionHandler_H 

#include "G4VExceptionHandler.hh"
#include "G4ExceptionSeverity.hh"

class RunManager;
class RunManagerMTMaster;
 
class ExceptionHandler : public G4VExceptionHandler
{
public:
    ExceptionHandler(RunManager * rm);
    ExceptionHandler(RunManagerMTMaster * rm);
    ExceptionHandler() {} ;
    virtual ~ExceptionHandler();
    int operator==(const ExceptionHandler & right) const { return (this == &right); }
    int operator!=(const ExceptionHandler & right) const { return (this != &right); }
    virtual bool Notify(const char * exceptionOrigin, const char * exceptionCode,
			G4ExceptionSeverity severity, const char * description);
private:
    ExceptionHandler(const ExceptionHandler &) : G4VExceptionHandler() {}
    ExceptionHandler& operator=(const ExceptionHandler &right) { return *this; }
    RunManager * fRunManager;
    RunManagerMTMaster * fRunManagerMT;
    //bool override;
    //int verbose;
};

#endif

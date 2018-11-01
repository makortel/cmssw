from __future__ import print_function
import six

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Mixins import _ConfigureComponent, _Labelable, _Parameterizable, _modifyParametersFromDict, PrintOptions, saveOrigin
from FWCore.ParameterSet.Modules import EDProducer
from FWCore.ParameterSet.SequenceTypes import _SequenceLeaf


# Eventually to be moved under FWCore/ParameterSet, but I want to
# avoid recompiling the universe for now

class SwitchProducer(EDProducer):
    """This class is to provide a switch of producers given a function making the decision.

    Intended to be inherited, and the inheriting class to pass the
    function. This way the objects can be pickled (for productionn)
    and edmConfigDumped (for debugging).
    """
    def __init__(self, availableResources, **kargs):
        super(SwitchProducer,self).__init__(None) # let's try None as the type...
        self._availableResourcesFunction = availableResources

        self.__setParameters(kargs)
        self._isModified = False

    def _getProducer(self):
        resources = self._availableResourcesFunction()
        for res in resources:
            if res in self.__dict__:
                return self.__dict__[res]
        raise RuntimeError("No implementation for any of the available resources: "+str(resources))

    # Mimick __Parameterizable
    #@staticmethod
    #def _raiseBadSetAttr(name):
    #    raise TypeError(name+" does not already exist, so it can only be set to a CMS python configuration type")

    def __addParameter(self, name, value):
        if not isinstance(value, cms.EDProducer):
            raise TypeError(name+" does not already exist, so it can only be set to a cms.EDProducer")
        if name in self.__dict__:
            message = "Duplicate insert of member " + name
            message += "\nThe original parameters are:\n"
            message += self.dumpPython() + '\n'
            raise ValueError(message)
        self.__dict__[name]=value
        self._Parameterizable__parameterNames.append(name)
        self._isModified = True

    def __setParameters(self, parameters):
        for name, value in six.iteritems(parameters):
            self.__addParameter(name, value)

    def __setattr__(self, name, value):
        # Following snippet copied from _Parameterizable in order to support Modifier.toModify
        #
        #since labels are not supposed to have underscores at the beginning
        # I will assume that if we have such then we are setting an internal variable
        if self.isFrozen() and not (name in ["_Labelable__label","_isFrozen"] or name.startswith('_')):
            message = "Object already added to a process. It is read only now\n"
            message +=  "    %s = %s" %(name, value)
            message += "\nThe original parameters are:\n"
            message += self.dumpPython() + '\n'
            raise ValueError(message)

        # underscored names bypass checking for _ParameterTypeBase
        if name[0]=='_':
            super(SwitchProducer, self).__setattr__(name,value)
        elif not name in self.__dict__:
            self.__addParameter(name, value)
            self._isModified = True
        else:
            # We should always receive an cms.EDProducer
            self.__dict__[name] = value
            self._isModified = True

    # Mimick _TypedParameterizable
    def clone(self, *args, **params):
        func = self._availableResourcesFunction
        if len(args) == 1:
            func = args[0]
        elif len(args) > 1:
            raise RuntimeError("SwitchProducer accepts at most one positional argument (for resource availability function")

        returnValue = SwitchProducer.__new__(type(self))

        # Need special treatment as cms.EDProducer is not a valid parameter type (except in this case)
        myparams = dict()
        for name, value in six.iteritems(params):
            if value is None:
                continue
            elif isinstance(value, dict):
                myparams[name] = self.__dict__[name].clone(**value)
            else: # value is an EDProducer
                myparams[name] = value.clone()

        # Add the ones that were not customized
        for name in self.parameterNames_():
            if name not in params:
                myparams[name] = self.__dict__[name].clone()

        returnValue.__init__(self._availableResourcesFunction, **myparams)
        returnValue._isModified = False
        returnValue._isFrozen = False
        saveOrigin(returnValue, 1)
        return returnValue

    def dumpPython(self, options=PrintOptions()):
        # What should we put for the first argument for the printout?
        # We don't know the function name
        # So the output of edmConfigDump does not necessarily reproduce the original configuration?
        # Pickling works
        # Could make the first argument to be a list as well, then we could dump the list returned by the function
        # But could that be confusing?
        result = "cms.%s(" % self.__class__.__name__
        options.indent()
        for resource in sorted(self.parameterNames_()):
            result += "\n" + options.indentation() + resource + " = " + getattr(self, resource).dumpPython(options).rstrip() + ","
        if result[-1] == ",":
            result = result.rstrip(",")
        options.unindent()
        result += "\n)\n"
        return result

    def nameInProcessDesc_(self, myname):
        return myname
    def moduleLabel_(self, myname):
        return myname
    def insertInto(self, parameterSet, myname):
        producer = self._getProducer()
        producer.insertInto(parameterSet, myname)

    # Mimick _Module
    def _clonesequence(self, lookuptable):
        try:
            return lookuptable[id(self)]
        except:
            raise ModuleCloneError(self._errorstr())
        def _errorstr(self):
            return "SwitchProducer" # TODO:



if __name__ == "__main__":
    import unittest

    # Copied from Config.py, try to avoid...
    class TestMakePSet(object):
        """Has same interface as the C++ object that creates PSets
        """
        def __init__(self):
            self.values = dict()
        def __insertValue(self,tracked,label,value):
            self.values[label]=(tracked,value)
        def addInt32(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVInt32(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addUInt32(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVUInt32(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addInt64(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVInt64(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addUInt64(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVUInt64(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addDouble(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVDouble(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addBool(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addString(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVString(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addInputTag(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVInputTag(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addESInputTag(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVESInputTag(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addEventID(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVEventID(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addLuminosityBlockID(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addLuminosityBlockID(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addEventRange(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVEventRange(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addPSet(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addVPSet(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addFileInPath(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def newPSet(self):
            return TestMakePSet()


    class testSwitchProducer(unittest.TestCase):
        def testConstruction(self):
            sp = SwitchProducer(lambda: [], cuda = cms.EDProducer("Foo"), cpu = cms.EDProducer("Bar"))
            self.assertEqual(sp.cuda.type_(), "Foo")
            self.assertEqual(sp.cpu.type_(), "Bar")
            #print(sp.dumpPython())
        def testGetProducer(self):
            sp = SwitchProducer(lambda: ["cpu"], cuda = cms.EDProducer("Foo"), cpu = cms.EDProducer("Bar"))
            self.assertEqual(sp._getProducer().type_(), "Bar")
            sp = SwitchProducer(lambda: ["cuda"], cuda = cms.EDProducer("Foo"), cpu = cms.EDProducer("Bar"))
            self.assertEqual(sp._getProducer().type_(), "Foo")
            sp = SwitchProducer(lambda: ["cpu", "cuda"], cuda = cms.EDProducer("Foo"), cpu = cms.EDProducer("Bar"))
            self.assertEqual(sp._getProducer().type_(), "Bar")
            sp = SwitchProducer(lambda: ["cuda", "cpu"], cuda = cms.EDProducer("Foo"), cpu = cms.EDProducer("Bar"))
            self.assertEqual(sp._getProducer().type_(), "Foo")
            sp = SwitchProducer(lambda: ["cuda", "cpu"], cpu = cms.EDProducer("Bar"))
            self.assertEqual(sp._getProducer().type_(), "Bar")
            sp = SwitchProducer(lambda: ["cuda"], cpu = cms.EDProducer("Bar"))
            self.assertRaises(RuntimeError, sp._getProducer)
        def testClone(self):
            sp = SwitchProducer(lambda: [], 
                                cuda = cms.EDProducer("Foo",
                                                      a = cms.int32(1),
                                                      b = cms.PSet(c = cms.int32(2))),
                                cpu = cms.EDProducer("Bar",
                                                     aa = cms.int32(11),
                                                     bb = cms.PSet(cc = cms.int32(12))))
            # Simple clone
            cl = sp.clone()
            self.assertEqual(cl.cuda.type_(), "Foo")
            self.assertEqual(cl.cuda.a.value(), 1)
            self.assertEqual(cl.cuda.b.c.value(), 2)
            self.assertEqual(cl.cpu.type_(), "Bar")
            self.assertEqual(cl.cpu.aa.value(), 11)
            self.assertEqual(cl.cpu.bb.cc.value(), 12)

            # Modify values with a dict
            cl = sp.clone(cuda = dict(a = 4, b = dict(c = None)),
                          cpu = dict(aa = 15, bb = dict(cc = 45, dd = cms.string("foo"))))
            self.assertEqual(cl.cuda.a.value(), 4)
            self.assertEqual(cl.cuda.b.hasParameter("c"), False)
            self.assertEqual(cl.cpu.aa.value(), 15)
            self.assertEqual(cl.cpu.bb.cc.value(), 45)
            self.assertEqual(cl.cpu.bb.dd.value(), "foo")

            # Replace/add/remove EDProducers
            cl = sp.clone(cuda = cms.EDProducer("Fred", x = cms.int32(42)),
                          fpga = cms.EDProducer("Wilma", y = cms.int32(24)),
                          cpu = None)
            self.assertEqual(cl.cuda.type_(), "Fred")
            self.assertEqual(cl.cuda.x.value(), 42)
            self.assertEqual(cl.fpga.type_(), "Wilma")
            self.assertEqual(cl.fpga.y.value(), 24)
            self.assertEqual(hasattr(cl, "cpu"), False)
        def testModify(self):
            sp = SwitchProducer(lambda: [],
                                cuda = cms.EDProducer("Foo",
                                                      a = cms.int32(1),
                                                      b = cms.PSet(c = cms.int32(2))),
                                cpu = cms.EDProducer("Bar",
                                                     aa = cms.int32(11),
                                                     bb = cms.PSet(cc = cms.int32(12))))
            m = cms.Modifier()
            m._setChosen()

            # Modify parameters
            m.toModify(sp,
                       cuda = dict(a = 4, b = dict(c = None)),
                       cpu = dict(aa = 15, bb = dict(cc = 45, dd = cms.string("foo"))))
            self.assertEqual(sp.cuda.a.value(), 4)
            self.assertEqual(sp.cuda.b.hasParameter("c"), False)
            self.assertEqual(sp.cpu.aa.value(), 15)
            self.assertEqual(sp.cpu.bb.cc.value(), 45)
            self.assertEqual(sp.cpu.bb.dd.value(), "foo")

            # The following etc doesn't work because of
            # AttributeError: 'EDProducer' object has no attribute 'setValue
            # from _modifyParametersFromDict
            #m.toModify(sp, cuda = cms.EDProducer("Xyzzy"))

            # Replace a producer
            m.toReplaceWith(sp.cuda, cms.EDProducer("Fred", x = cms.int32(42)))
            self.assertEqual(sp.cuda.type_(), "Fred")
            self.assertEqual(sp.cuda.x.value(), 42)

            # Add a producer
            # Currently gives "KeyError: 'Unknown parameter name fpga specified while calling Modifier'"
            #m.toModify(sp, fpga = cms.EDProducer("Wilma", y = cms.int32(24)))
            #self.assertEqual(sp.fpga.type_(), "Wilma")
            #self.assertEqual(sp.fpga.y.value(), 24)

            # Remove a producer
            m.toModify(sp, cpu = None)
            self.assertEqual(hasattr(sp, "cpu"), False)

        def testReplace(self):
            p = cms.EDProducer("Xyzzy", a = cms.int32(1))
            sp = SwitchProducer(lambda: [],
                                cuda = cms.EDProducer("Foo",
                                                      a = cms.int32(1),
                                                      b = cms.PSet(c = cms.int32(2))),
                                cpu = cms.EDProducer("Bar",
                                                     aa = cms.int32(11),
                                                     bb = cms.PSet(cc = cms.int32(12))))
            m = cms.Modifier()
            m._setChosen()

            # Currently fails with "TypeError: cpu does not already exist, so it can only be set to a CMS python configuration type"
            #m.toReplaceWith(p, sp)
            #self.assertEqual(isinstance(p, SwitchProducer), true)

            # Currently fails with "TypeError: toReplaceWith requires both arguments to be the same class type"
            p = cms.EDProducer("Xyzzy", a = cms.int32(1))
            #m.toReplaceWith(sp, p)
            #self.assertEqual(isinstance(sp, EDProducer), true)
            #self.assertEqual(isinstance(sp, SwitchProducer), false)

        def testDumpPython(self):
            sp = SwitchProducer(lambda: [],
                                cuda = cms.EDProducer("Foo",
                                                      a = cms.int32(1),
                                                      b = cms.PSet(c = cms.int32(2))),
                                cpu = cms.EDProducer("Bar",
                                                     aa = cms.int32(11),
                                                     bb = cms.PSet(cc = cms.int32(12))))
            self.assertEqual(sp.dumpPython(),
"""cms.SwitchProducer(
    cpu = cms.EDProducer("Bar",
        aa = cms.int32(11),
        bb = cms.PSet(
            cc = cms.int32(12)
        )
    ),
    cuda = cms.EDProducer("Foo",
        a = cms.int32(1),
        b = cms.PSet(
            c = cms.int32(2)
        )
    )
)
""")

        def testProcess(self):
            p = cms.Process("test")
            p.sp = SwitchProducer(lambda: ["cuda", "cpu"],
                                  cuda = cms.EDProducer("Foo",
                                                      a = cms.int32(1),
                                                      b = cms.PSet(c = cms.int32(2))),
                                  cpu = cms.EDProducer("Bar",
                                                     aa = cms.int32(11),
                                                     bb = cms.PSet(cc = cms.int32(12))))
            p.a = cms.EDProducer("A")
            p.s = cms.Sequence(p.a + p.sp)
            p.t = cms.Task(p.a, p.sp)
            p.p = cms.Path()
            p.p.associate(p.t)

            mp = TestMakePSet()
            p.fillProcessDesc(mp)
            self.assertEqual(mp.values["a"][1].values["@module_type"], (True, "A"))
            self.assertEqual(mp.values["sp"][1].values["@module_edm_type"], (True, "EDProducer"))
            self.assertEqual(mp.values["sp"][1].values["@module_type"], (True, "Foo"))
            self.assertEqual(mp.values["sp"][1].values["a"], (True, 1))

    unittest.main()

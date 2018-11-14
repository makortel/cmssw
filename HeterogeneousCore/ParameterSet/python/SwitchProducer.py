from __future__ import print_function
import six
import importlib

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.Mixins import _ConfigureComponent, _Labelable, _Parameterizable, _modifyParametersFromDict, PrintOptions, saveOrigin
from FWCore.ParameterSet.Modules import EDProducer
from FWCore.ParameterSet.SequenceTypes import _SequenceLeaf


# Eventually to be moved under FWCore/ParameterSet, but I want to
# avoid recompiling the universe for now

class SwitchProducer(EDProducer):
    """This class is to provide a switch of producers.
    """
    def __init__(self, **kargs):
        super(SwitchProducer,self).__init__(None) # let's try None as the type...

        self.__setParameters(kargs)
        self._isModified = False
        self._modulePath = "HeterogeneousCore.ParameterSet."

    def _chooseResource(self):
        #resources = ["cuda", "cpu"] # TODO: really implement...
        #for res in resources:
        #    if res in self.__dict__:
        #        return res

        cases = self.parameterNames_()
        bestCase = None
        for case in cases:
            mod = importlib.import_module(self._modulePath+case)
            if mod.enabled() and (bestCase is None or bestCase[0] < mod.priority()):
                bestCase = (mod.priority(), case)
        if bestCase is None:
            raise RuntimeError("All cases '%s' were disabled" % (str(cases)))
        return bestCase[1]

    def _getProducer(self):
        return self.__dict__[self._chooseResource()]

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
    def clone(self, **params):
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

        returnValue.__init__(**myparams)
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
    def appendToProcessDescList_(self, lst, myname):
        # This way we can insert the chosen EDProducer to @all_modules
        # so that we get easily a worker for it
        lst.append(myname)
        for case in self.parameterNames_():
            lst.append(myname+"@"+case)
    def insertInto(self, parameterSet, myname):
        for case in self.parameterNames_():
            producer = self.__dict__[case]
            producer.insertInto(parameterSet, myname+"@"+case)
        newpset = parameterSet.newPSet()
        newpset.addString(True, "@module_label", self.moduleLabel_(myname))
        newpset.addString(True, "@module_type", type(self).__name__)
        newpset.addString(True, "@module_edm_type", "EDProducer")
        newpset.addVString(True, "@all_cases", [myname+"@"+p for p in self.parameterNames_()])
        newpset.addString(False, "@chosen_case", myname+"@"+self._chooseResource())
        parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)

    def _placeImpl(self,name,proc):
        proc._placeProducer(name,self)
        for case in self.parameterNames_():
            # Note that these don't end up in @all_modules
            # automatically because they're not part of any
            # Task/Sequence/Path
            proc._placeProducer(name+"@"+case, self.__dict__[case])
        proc._placeAlias("@"+name, _SwitchProducerAlias(name, name+"@"+self._chooseResource()))

    # Mimick _Module
    def _clonesequence(self, lookuptable):
        try:
            return lookuptable[id(self)]
        except:
            raise ModuleCloneError(self._errorstr())
        def _errorstr(self):
            return "SwitchProducer" # TODO:

class _SwitchProducerAlias(object):
    def __init__(self, aliasFrom, aliasTo):
        super(_SwitchProducerAlias, self).__init__()
        self._aliasFrom = aliasFrom
        self._aliasTo = aliasTo

    def nameInProcessDesc_(self, myname):
        return myname;

    def appendToProcessDescList_(self, lst, myname):
        lst.append(self.nameInProcessDesc_(myname))

    def insertInto(self, parameterSet, myname):
        newpset = parameterSet.newPSet()
        newpset.addString(True, "@module_label", self._aliasFrom)
        newpset.addString(True, "@module_type", "SwitchProducer")
        newpset.addString(True, "@module_edm_type", "EDAlias")
        newpset.addString(False, "@alias_to", self._aliasTo)
        parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)

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
        def getString(self, tracked, label):
            elem = self.values[label]
            if elem[0] != tracked:
                raise Exception("%s: no such %s parameter" % (label, "tracked" if tracked else "untracked"))
            return elem[1]
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
        def getPSet(self, tracked, label):
            elem = self.values[label]
            if elem[0] != tracked:
                raise Exception("%s: no such %s parameter" % (label, "tracked" if tracked else "untracked"))
            return elem[1]
        def addVPSet(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def addFileInPath(self,tracked,label,value):
            self.__insertValue(tracked,label,value)
        def newPSet(self):
            return TestMakePSet()


    class testSwitchProducer(unittest.TestCase):
        def testConstruction(self):
            sp = SwitchProducer(cuda = cms.EDProducer("Foo"), cpu = cms.EDProducer("Bar"))
            self.assertEqual(sp.cuda.type_(), "Foo")
            self.assertEqual(sp.cpu.type_(), "Bar")
            #print(sp.dumpPython())
        def testGetProducer(self):
            sp = SwitchProducer(test1 = cms.EDProducer("Foo"), test2 = cms.EDProducer("Bar"))
            sp._modulePath += "test."
            self.assertEqual(sp._getProducer().type_(), "Bar")
            sp = SwitchProducer(test1dis = cms.EDProducer("Foo"), test2 = cms.EDProducer("Bar"))
            sp._modulePath += "test."
            self.assertEqual(sp._getProducer().type_(), "Bar")
            sp = SwitchProducer(test1 = cms.EDProducer("Foo"), test2dis = cms.EDProducer("Bar"))
            sp._modulePath += "test."
            self.assertEqual(sp._getProducer().type_(), "Foo")
            sp = SwitchProducer(test1 = cms.EDProducer("Bar"))
            sp._modulePath += "test."
            self.assertEqual(sp._getProducer().type_(), "Bar")
            sp = SwitchProducer(test1dis = cms.EDProducer("Bar"))
            sp._modulePath += "test."
            self.assertRaises(RuntimeError, sp._getProducer)
        def testClone(self):
            sp = SwitchProducer(cuda = cms.EDProducer("Foo",
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
            sp = SwitchProducer(cuda = cms.EDProducer("Foo",
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

            m.toModify(sp, cuda = cms.EDProducer("Xyzzy")) # Do we actually want to allow this? In a sense the toReplaceWith would be more logical

            # Replace a producer
            m.toReplaceWith(sp.cuda, cms.EDProducer("Fred", x = cms.int32(42)))
            self.assertEqual(sp.cuda.type_(), "Fred")
            self.assertEqual(sp.cuda.x.value(), 42)

            # Add a producer
            m.toModify(sp, fpga = cms.EDProducer("Wilma", y = cms.int32(24)))
            self.assertEqual(sp.fpga.type_(), "Wilma")
            self.assertEqual(sp.fpga.y.value(), 24)

            # Remove a producer
            m.toModify(sp, cpu = None)
            self.assertEqual(hasattr(sp, "cpu"), False)

        def testReplace(self):
            p = cms.EDProducer("Xyzzy", a = cms.int32(1))
            sp = SwitchProducer(cuda = cms.EDProducer("Foo",
                                                      a = cms.int32(1),
                                                      b = cms.PSet(c = cms.int32(2))),
                                cpu = cms.EDProducer("Bar",
                                                     aa = cms.int32(11),
                                                     bb = cms.PSet(cc = cms.int32(12))))
            m = cms.Modifier()
            m._setChosen()

            # Currently fails with "TypeError: cpu does not already exist, so it can only be set to a CMS python configuration type"
            # Fixing would require messing with the internals of _Parameterizable (to allow addition of EDProducer parameters),
            # maybe we don't want that? The use case can anyway be circumvented by always having p as a SwitchProducer, and
            # adding more cases with toModify()
            #m.toReplaceWith(p, sp)
            #self.assertEqual(isinstance(p, SwitchProducer), true)

            # Currently fails with "TypeError: toReplaceWith requires both arguments to be the same class type"
            # This might be easier than the previous case as we're messing with SwitchProducer, but with similar arguments the same
            # can be achieved by removing all cases but one with toModify()
            p = cms.EDProducer("Xyzzy", a = cms.int32(1))
            #m.toReplaceWith(sp, p)
            #self.assertEqual(isinstance(sp, EDProducer), true)
            #self.assertEqual(isinstance(sp, SwitchProducer), false)

        def testDumpPython(self):
            sp = SwitchProducer(test2 = cms.EDProducer("Foo",
                                                      a = cms.int32(1),
                                                      b = cms.PSet(c = cms.int32(2))),
                                test1 = cms.EDProducer("Bar",
                                                       aa = cms.int32(11),
                                                       bb = cms.PSet(cc = cms.int32(12))))
            self.assertEqual(sp.dumpPython(),
"""cms.SwitchProducer(
    test1 = cms.EDProducer("Bar",
        aa = cms.int32(11),
        bb = cms.PSet(
            cc = cms.int32(12)
        )
    ),
    test2 = cms.EDProducer("Foo",
        a = cms.int32(1),
        b = cms.PSet(
            c = cms.int32(2)
        )
    )
)
""")

        def testProcess(self):
            p = cms.Process("test")
            p.sp = SwitchProducer(test2 = cms.EDProducer("Foo",
                                                         a = cms.int32(1),
                                                         b = cms.PSet(c = cms.int32(2))),
                                  test1 = cms.EDProducer("Bar",
                                                         aa = cms.int32(11),
                                                         bb = cms.PSet(cc = cms.int32(12))))
            p.sp._modulePath += "test."
            p.a = cms.EDProducer("A")
            p.s = cms.Sequence(p.a + p.sp)
            p.t = cms.Task(p.a, p.sp)
            p.p = cms.Path()
            p.p.associate(p.t)

            mp = TestMakePSet()
            p.fillProcessDesc(mp)
            self.assertEqual(mp.values["a"][1].values["@module_type"], (True, "A"))
            self.assertEqual(mp.values["sp"][1].values["@module_edm_type"], (True, "EDProducer"))
            self.assertEqual(mp.values["sp"][1].values["@module_type"], (True, "SwitchProducer"))
            self.assertEqual(mp.values["@all_modules"][1], ["a", "sp", "sp@test2"])

    unittest.main()

import simpleml.baseclasses as bc


classes = ['BinaryClassifier']


def test_req_meth():
    for clsname in classes:
        cls = getattr(bc, clsname)

        def func(cls):
            for funcname in cls._req_meth:
                assert funcname in dir(cls)
        yield func, cls

class NotBinClass:  # pragma: no cover
    def fit(self): pass

class SimpBinClass(NotBinClass):  # pragma: no cover
    def classify(self): pass
    def test_err(self): pass

class TestSubclassBinClass:
    def test1(self):
        assert not issubclass(NotBinClass, bc.BinaryClassifier)

    def test2(self):
        assert issubclass(SimpBinClass, bc.BinaryClassifier)

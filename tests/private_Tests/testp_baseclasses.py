import simpleml.baseclasses as bc


classes = ['BinaryClassifier', 'BinaryClassifierWithErrors']


def test_req_meth():
    for clsname in classes:
        cls = getattr(bc, clsname)

        def func(cls):
            for funcname in cls._req_meth:
                getattr(cls, funcname)(None)
        yield func, cls

class NotBinClass:  # pragma: no cover
    def fit(self): pass

class SimpBinClass(NotBinClass):  # pragma: no cover
    def classify(self): pass

class SimpBinClassWithErr(SimpBinClass):  # pragma: no cover
    def train_err(self): pass
    def test_err(self): pass

class TestSubclassBinClass:
    def test1(self):
        assert not issubclass(NotBinClass, bc.BinaryClassifier)

    def test2(self):
        assert issubclass(SimpBinClass, bc.BinaryClassifier)

    def test3(self):
        assert issubclass(SimpBinClassWithErr, bc.BinaryClassifier)

class TestSubclassBinClassWithErr:
    def test1(self):
        assert not issubclass(NotBinClass, bc.BinaryClassifierWithErrors)

    def test2(self):
        assert not issubclass(SimpBinClass, bc.BinaryClassifierWithErrors)

    def test3(self):
        assert issubclass(SimpBinClassWithErr, bc.BinaryClassifierWithErrors)

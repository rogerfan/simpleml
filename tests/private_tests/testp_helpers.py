import numpy as np

import simpleml.helpers as helpers


class TestNPPrintOptions:

    def test_print(self):
        with helpers.np_print_options(precision=2):
            print(np.array([1.234234, 2.200003]))

    def test_precision(self):
        with helpers.np_print_options(precision=2):
            res = str(np.array([1.234234, 2.200003]))

        assert res == '[ 1.23  2.20]'

    def test_strip_zeros(self):
        with helpers.np_print_options(precision=3):
            res = str(np.array([1.200004, 2.200002]))

        assert res == '[ 1.2  2.2]'

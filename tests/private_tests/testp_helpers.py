import numpy as np

import simpleml.helpers as helpers


class TestNPPrintOptions:
    def test_print(self):
        with helpers.np_print_options(precision=2):
            print(np.array([1.234234, 2.200003]))

    def test_precision(self):
        with helpers.np_print_options(precision=2):
            res = str(np.array([1.234234, 2.200003]))

        print(res)
        assert res == '[ 1.23  2.2 ]'

    def test_strip_zeros(self):
        with helpers.np_print_options(strip_zeros=False, precision=2):
            res = str(np.array([1.234234, 2.200003]))

        print(res)
        assert res == '[ 1.23  2.20]'

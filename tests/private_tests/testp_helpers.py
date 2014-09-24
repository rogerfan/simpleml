import numpy as np
from nose.tools import raises

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


class TestCheckRandomState:
    def test_none(self):
        state = helpers.check_random_state(None)
        assert isinstance(state, np.random.RandomState)

    def test_int(self):
        state = helpers.check_random_state(5435)
        assert isinstance(state, np.random.RandomState)

    def test_state(self):
        orig_state = np.random.RandomState()
        state = helpers.check_random_state(orig_state)
        assert isinstance(state, np.random.RandomState)
        assert state is orig_state

    @raises(ValueError)
    def test_invalid(self):
        helpers.check_random_state(4.35)

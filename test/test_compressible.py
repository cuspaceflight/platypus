import pytest

from platypus import compressible as comp

def test_1d():
    assert abs(comp.calcLimitMassFlow(1.4) - 1.281)<0.001
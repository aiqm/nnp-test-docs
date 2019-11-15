"""
Use ASE to Simulate Planetary Motion
====================================

This tutorial shows how to use `ASE`_ to simulate planetary motion.
Although both ASE and NNP are designed for molecular dynamics (MD),
implementing a force field to do MD is not at a reasonable abount of
work to use as a tutorial. So we simulate something different here.
Assume we have two atoms, one hydrogen and one carbon. They are all
moving under the gravity centered at the origin. There is no interaction
between these two atoms.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

###############################################################################
# Let's first import all the packages we will use.
import torch
import ase
import nnp
import pytest


###############################################################################
# Let's first import all the packages we will use.

def test_result():
    return

if __name__ == '__main__':
    pytest.main()

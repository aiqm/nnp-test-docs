"""
Analytical Hessians and Vibrational Analysis
============================================

This tutorial demonstrates how compute analytical hessians and do
vibrational analysis using ``nnp.vib``.
"""
###############################################################################
# Let's first import all the packages we will use:
import torch
import math
import pytest
import sys
import nnp.so3 as so3
import nnp.vib as vib


###############################################################################
# In this tutorial, we will study atoms moving in a quadratic potential.
# There is no interaction between these atoms.
#
# Let's first construct such a potential. A naive potential would be:
#
# .. math::
#   U(x, y, z) = \frac{1}{2} \left(0.5 x^2 + y^2 + 2 z^2\right)
def naive_potential(x, y, z):
    return 0.5 * (0.5 * x ** 2 + y ** 2 + 2 * z ** 2)


###############################################################################
# A naive potential is not very interesting, let's rotate the potential along
# the z axis for 45 degrees. Rotating the potential along the z axis for 45 degrees
# is equivalent to rotate the coordinates along the z axis for -45 degrees before
# evaluating the naive potential.  Let's make our potential able to handle both
# single molecule (i.e. coordinates has shape `(atoms, 3)`), and in batch (i.e.
# coordinates has shape `(molecules, atoms, 3)`):
rot45 = so3.rotate_along(torch.tensor([0, 0, math.pi / 4]))
rot_neg_45 = so3.rotate_along(torch.tensor([0, 0, -math.pi / 4]))


def potential(coordinates):
    rotated_coordinates = (rot_neg_45 @ coordinates.transpose(-1, -2)).transpose(-1, -2)
    x, y, z = rotated_coordinates.unbind(-1)
    return naive_potential(x, y, z).sum(dim=-1)


###############################################################################
# Now let's compute the hessian one molecule containing two atoms at the origin
coordinates = torch.zeros(2, 3, requires_grad=True)
energy = potential(coordinates)
hessian = vib.hessian(coordinates, energies=energy)
print(hessian)

###############################################################################
# Let's compute the theoretical result to see if it matches with the result above:
#
# First of all, because there are no interactions between atoms, the hessian
# should be a block diagonal matrix:
#
# .. math::
#   H = \left[9\times9\right] = \left[\begin{array}{cc}
#                                     3\times3\\
#                                     & 3\times3
#                                     \end{array}\right]
#
# The two :math:`3\times3` matricies are identical. For each :math:`3\times3`
# matrix, the potential is not rotated at the z axis, so the structure of it
# should be:
#
# .. math::
#   \left[\begin{array}{ccc}
#    ? & ?\\
#    ? & ?\\
#    &  & 2
#    \end{array}\right]
#
# It is not hard to figure out that, for the rotated potential, considering
# only the contribution from x,y plane, it can be written as:
#
# .. math::
#   U=\frac{1}{2}\left[0.5\left(\frac{x+y}{\sqrt{2}}\right)^{2}+\left(\frac{y-x}{\sqrt{2}}\right)^{2}\right]=\frac{1}{4}\left(1.5x^{2}+1.5y^{2}-xy\right)
#
# Therefore, the :math:`2\times2` block on the top left should be
#
# .. math::
#   \left[\begin{array}{cc}
#    0.75 & -0.25\\
#    -0.25 & 0.75
#    \end{array}\right]
def test_analytical_hessian():
    hessian00 = hessian[:3, :3]
    hessian01 = hessian[:3, 3:]
    hessian10 = hessian[3:, :3]
    hessian11 = hessian[3:, 3:]
    expected = torch.tensor([
        [ 0.75, -0.25, 0],  # noqa: E201, E241
        [-0.25,  0.75, 0],  # noqa: E201, E241
        [ 0.00,  0.00, 2],  # noqa: E201, E241
    ])
    assert torch.allclose(hessian00, expected)
    assert torch.allclose(hessian11, expected)
    assert torch.allclose(hessian10, torch.zeros(3, 3))
    assert torch.allclose(hessian01, torch.zeros(3, 3))


###############################################################################
# Now let's run all the tests
if __name__ == '__main__':
    pytest.main([sys.argv[0], '-v'])
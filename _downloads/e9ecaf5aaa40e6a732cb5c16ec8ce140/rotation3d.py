"""
Rotation in 3D Space
====================

This tutorial demonstrates how rotate a group of points in 3D space.
"""
###############################################################################
# To begin with, we must understand that rotation is a linear transformation.
# The set of all rotations forms the group SO(3). It is a Lie group that each
# group element is described by an orthogonal 3x3 matrix.
#
# That is, if you have two points :math:`\vec{r}_1` and :math:`\vec{r}_2`, and
# you want to rotate the two points along the same axis for the same number of
# degrees, then there is a single orthogonal matrix :math:`R` that no matter
# the value of :math:`\vec{r}_1` and :math:`\vec{r}_2`, their rotation is always
# :math:`R\cdot\vec{r}_1` and :math:`R\cdot\vec{r}_2`.
#
# Let's import libraries first
import math
import torch
import nnp.so3 as so3

###############################################################################
# Let's first take a look at a special case: rotating ``(1, 0, 0)``, ``(0, 1, 0)``
# and ``(0, 0, 1)`` along the diagonal for 120 degree, this should permute these
# points:
axis = torch.ones(3) / math.sqrt(3) * (math.pi * 2 / 3)
R = so3.rotate_along(axis)
points = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=torch.float)
rotated = R @ points
print(R)

print(R.t() @ R)

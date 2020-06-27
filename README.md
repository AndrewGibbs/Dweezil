# Dweezil
Designed for problems of acoustic scattering by screens of a certain class, which permit a mesh that can be embedded inside of a large uniform mesh on a parallelogram. Used with [bempp-cl](https://github.com/bempp/bempp-cl)

Dweezil works by creating a series of problems on sub-meshes of the rectangular mesh, each requiring only O(N) memory. A Block-Block-Toeplitz-Toeplitz-Block (BBTTB) matrix is constructed, requiring O(N) memory, which can be inverted in O(N\log N) FLOPS using elementary techniques.

In terms of CPU time, this construction of the BBTTB matrix is far from optimal, however there is a significant memory saving compared the O(N^2) memory required for standard BEM.

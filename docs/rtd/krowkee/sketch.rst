.. _krowkee-sketch:

`krowkee::sketch` module reference.
===================================

`krowkee::sketch::Sketch` is the basic interface for specifiying sketch data
structures in `krowkee`.
However, the behavior of `krowkee::sketch::Sketch` is heavily modfied by its
template arguments.
The `ContainerType` template determines the underlying memory managment behavior
of the register set.
`krowkee::sketch::Dense` yields the simplest behavior, and stores the registers
as an `std::vector`.
`krowkee::sketch::Sparse` is more sophisticated, and stores the registers as a
`krowkee::container::compacting_map` under the hood.
`krowkee::sketch::Promotable` marries the two, and allows a sketch to begin life
as with a Sparse container that is promoted to Dense if it becomes sufficiently
full.

.. toctree::
   :maxdepth: 2
   :caption: Register Container Types:

   sketch/Dense
   sketch/Sparse
   sketch/Promotable
   sketch/Matrix

A sketch is also defined by the transform that it supports.
This transform defines the way in which updates affect the set of registers, and
is fundamental to the statistical guarantees of the associated sketch.

.. toctree::
   :maxdepth: 2
   :caption: Transform Types:

   transform/SparseJLT
   transform/DoubleSparseJLT
   transform/FWHT

.. doxygenclass:: krowkee::sketch::Sketch

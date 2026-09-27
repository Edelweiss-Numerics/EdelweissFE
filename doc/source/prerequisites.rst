Prerequisites
#############

EdelweissFE requires a conda installation, e.g. `Miniforge <https://conda-forge.org/download/>`_.
All required packages, including the free-threaded Python interpreter, are defined in ``environment.yml`` in the
repository root:

.. literalinclude:: ../../environment.yml
   :language: yaml

Optionally,

* `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_ provides additional elements and constitutive models,
* Intel MKL (conda packages ``mkl`` and ``mkl-include``, Linux only) enables the PARDISO direct solver.

See :doc:`installation` for the installation steps.

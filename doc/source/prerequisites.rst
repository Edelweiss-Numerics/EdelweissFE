Prerequisites
#############

EdelweissFE requires a conda installation (e.g. `Miniforge <https://conda-forge.org/download/>`_) and
`conda-lock <https://conda.github.io/conda-lock/>`_. The complete environment, including the free-threaded Python
interpreter, is declared in ``environment.yml`` in the repository root and pinned by ``conda-lock.yml``:

.. literalinclude:: ../../environment.yml
   :language: yaml

Optionally, `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_ provides additional elements and
constitutive models.

See :doc:`installation` for the installation steps.

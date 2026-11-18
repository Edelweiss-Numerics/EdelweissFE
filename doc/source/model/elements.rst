Elements
========

Relevant module: ``edelweissfe.config.elementlibrary``

.. automodule:: edelweissfe.config.elementlibrary
   :members:

Provider ``marmot``
-------------------

Relevant module: ``edelweissfe.elements.marmotelement.element``

.. autoclass:: edelweissfe.elements.marmotelement.element.MarmotElementWrapper
   :members:

Heat conduction elements
^^^^^^^^^^^^^^^^^^^^^^^^

Marmot's transport elements ``DC2D3T``, ``DC2D4T``, ``DC2D8T`` (plane, ``type=plane`` section with a thickness),
``DCAX4T``, ``DCAX8T`` (axisymmetric, ``domain=axisymmetric``, :math:`r = x`; a plane section, whose thickness they do
not use) and ``DC3D4T``, ``DC3D8T``, ``DC3D20T`` (solid) carry the scalar field ``temperature`` and take a transport
material such as ``HeatConduction``. They integrate transient heat conduction with the backward Euler method over the
time increment of the step; a material with zero heat capacity gives steady state conduction. Loads:

- ``distributedload`` of ``type=heat flux`` (``magnitude=q``, into the body), ``type=heat film``
  (``magnitude='h, T_inf'``, convection) and ``type=heat radiation`` (``magnitude='c, T_inf, T_0'``,
  :math:`c = \varepsilon\sigma`, :math:`T_0` the absolute zero of the temperature scale). The amplitude scales all
  values: use ``f(t)='1'`` to apply them at once.
- ``bodyforce`` with ``forceVector`` the volumetric heat source.
- The initial temperature is set with ``setfield`` on a ``perNode`` field output of the result ``U`` in the first step.

A zero time increment (such as the one a step begins with) lets no heat flow. The heat flow at prescribed temperatures
is the result ``P`` of the field ``temperature``, in W (through the thickness of plane elements, over the full ring of
axisymmetric ones).

.. literalinclude:: ../../../testfiles/marmot/HeatTransientDC2D4T/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/HeatTransientDC2D4T/test.inp``

Moisture diffusion elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same eight element shapes with the suffix ``H`` (``DC2D3H``, ``DC2D4H``, ``DC2D8H``, ``DCAX4H``, ``DCAX8H``,
``DC3D4H``, ``DC3D8H``, ``DC3D20H``) carry the scalar field ``relative humidity`` (pore relative humidity, 0 ... 1) and
take the transport material ``MoistureDiffusion`` (Bažant–Najjar diffusivity, linear or BSB sorption isotherm,
optional self-desiccation). Loads:

- ``distributedload`` of ``type=moisture flux`` (``magnitude=q``, into the body) and ``type=moisture film``
  (``magnitude='beta, h_inf'``, exchange with an environment at the relative humidity ``h_inf``). As for the heat film,
  the amplitude scales both values: use ``f(t)='1'`` to expose a surface at once.
- ``bodyforce`` with ``forceVector`` a volumetric moisture source.
- The initial relative humidity is set with ``setfield`` on a ``perNode`` field output of the result ``U``.

The field ``relative humidity`` is converged when the correction is below 1e-8 (dimensionless) and the flux residual
below 1e-8 of the averaged element flux, or below the absolute floor 1e-14. The flux unit depends on the units of the
material (e.g. kg/(m² s), or m/s in the classical form with a unit moisture capacity): where the element fluxes cancel
(e.g. a sealed, uniformly self-desiccating body), raise the floor with
``*updateConfiguration, configuration=absoluteFluxResidualTolerance`` to a value fitting the units.

.. literalinclude:: ../../../testfiles/marmot/MoistureDryingDC2D8H/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/MoistureDryingDC2D8H/test.inp``

Coupled heat and moisture transport elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same eight element shapes with the suffix ``TH`` (``DC2D3TH``, ``DC2D4TH``, ``DC2D8TH``, ``DCAX4TH``, ``DCAX8TH``,
``DC3D4TH``, ``DC3D8TH``, ``DC3D20TH``) carry two fields per node, ``temperature`` and ``relative humidity`` (in this
order), and take the two-field transport material ``ThermoHygro`` (Bažant–Thonguthai-type coupling: Arrhenius
diffusivity, hygrothermic coefficient, vapour pressure driven moisture flux, heat capacity of the water, moisture
dependent conductivity, latent heat carried by the vapour, heat of hydration and self-desiccation over the equivalent
age). With 21 properties (no coupling) it reproduces ``HeatConduction`` and ``MoistureDiffusion``. All heat and
moisture loads act on their field; in addition:

- ``distributedload`` of ``type=latent heat film`` (``magnitude='c, h_inf'``): the heat taken away by the evaporating
  water, :math:`q_n = c\,(h - h_\infty)` on the temperature, with :math:`c = L_v\beta` (latent heat times the
  coefficient of the moisture film).
- ``bodyforce`` with ``forceVector='heat source, moisture source'``.

Both fields keep their tolerances. A sealed body without loads (e.g. adiabatic early-age concrete) needs units-aware
absolute flux floors for both fields (see the moisture elements).

.. literalinclude:: ../../../testfiles/marmot/ThermoHygroHeatedDryingDC2D8TH/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/ThermoHygroHeatedDryingDC2D8TH/test.inp``

Provider ``marmotsingleqpelement``
----------------------------------

Relevant module: ``edelweissfe.elements.marmotsingleqpelement.element``

.. autoclass:: edelweissfe.elements.marmotsingleqpelement.element.MarmotMaterialWrappingElement
   :members:

.. literalinclude:: ../../../testfiles/marmot/QPMarmotMaterialHypoElastic/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/QPMarmotMaterialHypoElastic/test.inp``

.. _provider_edelweiss_elements:

Provider ``edelweiss``
----------------------

The element type definition has to be of the following form:

**Elements**

One of the following element types needs to be included in the definition (``type`` in .inp file).

- **CPE4**    - quadrilateral 2D element with 4 nodes and plane strain.
- **CPE8**    - quadrilateral 2D element with 8 nodes and plane strain.
- **CPS4**    - quadrilateral 2D element with 4 nodes and plane stress.
- **CPS8**    - quadrilateral 2D element with 8 nodes and plane stress.
- **C3D8**    - hexahedron 3D element with 8 nodes.
- **C3D20**    - hexahedron 3D element with 20 nodes.

.. note::
    ``C3D8``/``C3D20`` (and their ``TL`` counterparts) use the standard Abaqus node ordering
    (corner ring 1-4 at :math:`\zeta=-1`, ring 5-8 at :math:`\zeta=+1`), matching the ``marmot``
    provider's elements -- there is only one node-ordering convention across the framework.

**additional Parameters**

The following optional Parameters are also included in the element type definition:

- **R**     - reduced integration for element, at the end of elementType.
- **E**     - extended integration for element, at the end of elementType.
- **N**     - (optional) regular integration, at the end of elementType.
- **TL**    - use the total Lagrangian element, before integration type.

Common base
~~~~~~~~~~~

Both elements below share everything that does not depend on the kinematics -- geometry and quadrature,
material and quadrature-point state, loads, mass and result access -- in a common base class. Each
element adds only how strain and stress follow from the nodal displacements.

Relevant module: ``edelweissfe.elements.base.displacementelementbase``

.. autoclass:: edelweissfe.elements.base.displacementelementbase.DisplacementElementBase
   :members:

Geometrically linear element
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This element uses a geometrically linear approach for the computation.

Relevant module: ``edelweissfe.elements.displacementelement.element``

.. autoclass:: edelweissfe.elements.displacementelement.element.DisplacementElement
   :members:

.. note::
    If R or E is not given by the user, we assume regular integration.

Example of the usage of a quadrilateral element with 4 nodes, regular integration and plane strain.

.. literalinclude:: ../../../examples/CantileverBeam/test4.inp
    :language: edelweiss
    :caption: Example: ``examples/CantileverBeam/test4.inp``

Example of the usage of a quadrilateral element with 8 nodes, regular integration and plane strain for the same example.

.. literalinclude:: ../../../examples/CantileverBeam/test8.inp
    :language: edelweiss
    :caption: Example: ``examples/CantileverBeam/test8.inp``

Example of the usage of a 3D hexahedron element with 8 nodes.

.. literalinclude:: ../../../examples/CantileverBeam/test3D.inp
    :language: edelweiss
    :caption: Example: ``examples/CantileverBeam/test3D.inp``

.. note::
    This element is only compatible with the material provider ``edelweiss``!

.. note::
    This element works with the Cauchy stress :math:`\sigma` and the tangent :math:`\frac{d\sigma}{d\varepsilon}`, where
    :math:`\varepsilon` stands for the linearized strain tensor!

Total Lagrange element
~~~~~~~~~~~~~~~~~~~~~~

This element is geometrically nonlinear and uses the total Lagrange algorithm for finite strain.

Relevant module: ``edelweissfe.elements.displacementtlelement.element``

.. autoclass:: edelweissfe.elements.displacementtlelement.element.DisplacementTLElement
    :members:

.. note::
    This element is only compatible with the material provider ``edelweiss``!

.. note::
    For hyperelastic materials, this element works with the second Piola-Kirchhoff stress :math:`\mathbf{S}` and the tangent
    :math:`\frac{d\mathbf{S}}{d\mathbf{E}}`, where :math:`\mathbf{E}` stands for the Green-Lagrange strain tensor!

    For other materials, this element works with the Kirchhoff stress :math:`\tau` and the tangent
    :math:`\frac{d\tau}{d\mathbf{F}}`, where :math:`\mathbf{F}` stands for the deformation gradient!

Contact facet elements
----------------------

Relevant module: ``edelweissfe.elements.contactsurfaceelement``

Thin, material-less, volume-less elements (``Tria3ContactFacet``/``Line2ContactFacet``) sharing the
ordinary displacement DOFs of a deformable body's boundary nodes, used as the master side of
:doc:`node-to-deformable-surface contact </model/constraints>`. Not user-constructed directly -- see the
:doc:`surfaceElementGenerator </model/generators>` generator.

Because a facet has neither a material nor a state, its kernels, its internal energy and its state
acceptance are all no-ops, and it reports ``hasKernels = False``. The model leaves such an entity out of its
discrete system (:ref:`discrete system <model-discrete-system>`): the facet has no block in the system matrix and no
location, and no solver calls into it on any increment. Its nodes have rows only through the elements that carry
kernels: a node that only a facet touches is numbered but has no row, and the discrete system refuses it, naming the
node. A load on an element set that holds facets, e.g. a body force on ``all``, acts on the elements of the set that
are located.
An element of your own needs to do nothing here: :class:`~edelweissfe.elements.base.baseelement.BaseElement`
reports ``True``, which is right for any element that carries a material.

.. automodule:: edelweissfe.elements.contactsurfaceelement
    :members: __doc__

.. autoclass:: edelweissfe.elements.contactsurfaceelement.Tria3ContactFacet
   :members:

.. autoclass:: edelweissfe.elements.contactsurfaceelement.Line2ContactFacet
   :members:

Implementing your own elements
------------------------------

Relevant module: ``edelweissfe.elements.base.baseelement``

An element follows the rule every entity of the discrete system follows (see
:ref:`residual-and-tangent-convention`): each method adds a residual contribution ``P`` and its tangent
``K = dP/dU``. ``computeKernels(K, P, U, dU, time, dT)`` adds the internal forces and the stiffness matrix -- the
tangent comes first in its arguments; the load methods ``computeDistributedLoad`` and ``computeBodyForce`` add the
*negative* of the external load, and its tangent if the load follows the deformation.

An element carrying a material should also implement
:meth:`~edelweissfe.elements.base.baseelement.BaseElement.updateMaterialProperty`, through which the
:doc:`changeMaterialProperty </analysis/stepactions>` step action changes a material property during an
analysis. Unlike ``setMaterial``, it must keep the element's state: neither reset nor re-initialize the
state variables, and keep the arrays holding them, which field outputs keep viewing.

.. automodule:: edelweissfe.elements.base.baseelement
   :members:
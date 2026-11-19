Keyword reference
=================

Every top-level keyword of the input file with its options. The syntax of an input file is described in
:doc:`syntax`. The tables are generated from each keyword's option schema in ``edelweissfe/keywords/``, the same
schema the parser validates an input file against; ``edelweissfe --keywords`` prints the same information.

Many keywords only select a type (``type=``, ``solver=``, ``generator=``, ...); the options of that type are
documented on the page linked below the keyword.

Model
*****

``*node`` - Define nodes
------------------------

See :doc:`model/mesh`.

.. pprint:: keyword:node

``*element`` - Define elements
------------------------------

See :doc:`model/mesh` and :doc:`model/elements`.

.. pprint:: keyword:element

``*elSet`` - Define an element set
----------------------------------

See :doc:`model/sets`.

.. pprint:: keyword:elSet

``*nSet`` - Define a node set
-----------------------------

See :doc:`model/sets`.

.. pprint:: keyword:nSet

``*surface`` - Define a surface
-------------------------------

See :doc:`model/sets`.

.. pprint:: keyword:surface

``*modelGenerator`` - Generate a mesh or model
----------------------------------------------

Generators and their options: :doc:`model/generators`.

.. pprint:: keyword:modelGenerator

``*material`` - Define a material
---------------------------------

Materials, providers and their properties: :doc:`model/materials`.

.. pprint:: keyword:material

``*advancedmaterial`` - Define a material by named options
----------------------------------------------------------

See :doc:`model/materials`.

.. pprint:: keyword:advancedmaterial

``*section`` - Assign a material to elements
--------------------------------------------

Section types and their options: :doc:`model/sections`.

.. pprint:: keyword:section

``*elementProperty`` - Override an element property
----------------------------------------------------

Sets the named property on all elements of an element set; the data lines hold its values.

.. pprint:: keyword:elementProperty

``*analyticalField`` - Define an analytical field
-------------------------------------------------

Field types and their options: :doc:`model/analyticalfields`.

.. pprint:: keyword:analyticalField

``*constraint`` - Define a constraint
-------------------------------------

Constraint types and their options: :doc:`model/constraints`.

.. pprint:: keyword:constraint

Analysis
********

``*job`` - Define the analysis job
----------------------------------

.. pprint:: keyword:job

``*solver`` - Define a nonlinear solver
---------------------------------------

Solvers and their options: :doc:`analysis/solvers`.

.. pprint:: keyword:solver

``*linearSolver`` - Define a linear solver
------------------------------------------

Linear solvers and their options: :doc:`analysis/linsolvers`.

.. pprint:: keyword:linearSolver

``*step`` - Define an analysis step
-----------------------------------

The data lines hold the options of the step type (e.g. ``maxInc``, ``maxNumInc``, ``stepLength``), see
:doc:`analysis/steps`; the ``>>`` blocks inside a step are the step actions, see :doc:`analysis/stepactions`.

.. pprint:: keyword:step

``*modelModifier`` - Modify the model during the analysis
---------------------------------------------------------

Model modifiers and their options: :doc:`analysis/modelmodifiers`.

.. pprint:: keyword:modelModifier

``*updateConfiguration`` - Change a configuration
-------------------------------------------------

Changes the settings of a configuration category, given as ``key=value`` data lines, e.g. the solver tolerances
(see :doc:`analysis/solvers`).

.. pprint:: keyword:updateConfiguration

``*restart`` - Resume from a checkpoint
---------------------------------------

.. pprint:: keyword:restart

Output
******

``*fieldOutput`` - Define field outputs
---------------------------------------

See :doc:`output/fieldoutput`.

.. pprint:: keyword:fieldOutput

``*output`` - Define an output manager
--------------------------------------

Output managers and their options: :doc:`output/outputmanagers`.

.. pprint:: keyword:output

``*configurePlots`` - Configure figures and axes
------------------------------------------------

See :doc:`output/plotter`.

.. pprint:: keyword:configurePlots

``*exportPlots`` - Export figures
---------------------------------

See :doc:`output/plotter`.

.. pprint:: keyword:exportPlots

Input files
***********

``*include`` - Include another input file
-----------------------------------------

.. pprint:: keyword:include

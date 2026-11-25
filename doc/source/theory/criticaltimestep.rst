Theory: the critical time step of an explicit analysis
======================================================

An explicit analysis (``NED``, :doc:`/analysis/solvers`) solves no system of equations: it divides the net force by a
lumped mass and steps on. The price is that it is stable only below a time increment, the *critical time step*. This
page explains where that limit comes from, why EdelweissFE estimates it instead of computing it, how the estimates of
the parts of a model are combined, what penalty constraints add to them, what the estimates do not see, and when each
estimate is taken anew. The code is :mod:`~edelweissfe.solvers.base.criticaltimestep`.

Where the limit comes from
--------------------------

Linearize the equation of motion about the current state, :math:`\boldsymbol{M}\ddot{\boldsymbol{u}} +
\boldsymbol{C}\dot{\boldsymbol{u}} + \boldsymbol{K}\boldsymbol{u} = \boldsymbol{0}`, with the lumped (diagonal) mass
:math:`\boldsymbol{M}`, damping :math:`\boldsymbol{C}` and the tangent stiffness :math:`\boldsymbol{K}`. In the
eigenvectors of :math:`\boldsymbol{M}^{-1}\boldsymbol{K}` it falls apart into single oscillators
:math:`\ddot{q} + \omega^2 q = 0`. The central difference method applied to one of them,

.. math::

   q_{n+1} - 2 q_n + q_{n-1} = -\omega^2 \Delta t^2 q_n ,

has the amplification factors :math:`z` with :math:`z^2 - (2 - \omega^2\Delta t^2) z + 1 = 0`. Their product is one;
they stay on the unit circle -- the oscillation neither grows nor decays -- as long as :math:`|2 - \omega^2 \Delta t^2|
\le 2`, i.e. :math:`\omega \Delta t \le 2`. Beyond, one of them is real and larger than one, and that mode grows
without bound. The whole system is stable if its highest mode is:

.. math::

   \Delta t \le \Delta t_\mathrm{crit} = \frac{2}{\omega_\mathrm{max}}, \qquad
   \omega_\mathrm{max}^2 = \lambda_\mathrm{max}(\boldsymbol{M}^{-1}\boldsymbol{K}) .

The damping does not lower this limit in ``NED``, which takes the damping force with the mean of the velocities before
and after the update. A field with a damping but no inertia (a first-order nonlocal field) is integrated by forward
Euler, :math:`\boldsymbol{C}\dot{\boldsymbol{u}} + \boldsymbol{K}\boldsymbol{u} = \boldsymbol{0}`, stable for
:math:`\Delta t \le 2/\lambda_\mathrm{max}(\boldsymbol{C}^{-1}\boldsymbol{K})`.

Why bounds, not eigenvalues
---------------------------

:math:`\omega_\mathrm{max}` of the assembled system is one eigenvalue of a matrix the explicit method otherwise never
assembles. Computing it -- by a few power or Lanczos iterations on the assembled tangent -- costs more than many
increments, and it changes as the state changes. What the method needs is not the eigenvalue but a time increment that
is *safe*: a lower bound of :math:`\Delta t_\mathrm{crit}`, from an upper bound of :math:`\omega_\mathrm{max}`, at a
cost small against an increment. Two such bounds are used:

* **The element eigenvalue inequality.** The highest frequency of an assembled mesh is not larger than the highest
  frequency of its elements, each taken alone with its own lumped mass:
  :math:`\omega_\mathrm{max}^2 \le \max_e \lambda_\mathrm{max}(\boldsymbol{M}_e^{-1}\boldsymbol{K}_e)` (the Rayleigh
  quotient of the mesh is a weighted mean of the elements' quotients). An element's eigenvalue problem is small, and
  Marmot bounds it with the element's own strain operator and the largest wave speed of its material, which holds for
  every element order and distortion of the geometry it is taken on -- where the classic "element length over wave
  speed" is exact only for a linear bar. A finite-strain element of Marmot takes its reference geometry, with the
  wave speed at its current deformation: for a strongly deformed element, that no longer bounds its eigenvalue, and
  the elements' policy rests on an assumption (see "When an estimate is taken anew").
* **Row sums.** The particles of a meshfree model (RKPM) have no such inequality: the stiffness of a particle couples
  nodes at which its own lumped mass vanishes. Every eigenvalue :math:`\lambda` of a matrix is bounded by its largest
  absolute row sum (Gershgorin), and more sharply by the Collatz-Wielandt bound of its absolute value with positive
  weights. EdelweissMeshfree applies it to the mass-scaled stiffness :math:`\boldsymbol{M}^{-1/2}\boldsymbol{K}
  \boldsymbol{M}^{-1/2}` with weights per field -- or, with ``carry-critical-time-step-weights``, with weights carried
  from estimate to estimate and refined by one step of the power iteration each time
  (:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfEntitiesWithOwnInertia`, "Carried weights")
  --, and Marmot bounds the row sums of every particle without assembling its stiffness (EdelweissMeshfree,
  ``MPMDiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia``).

Both are bounds from above for :math:`\omega_\mathrm{max}`, so the time increment they give is safe; how far below the
true limit it lies is the price of not computing the eigenvalue.

**Words.** A *bound* is on the side of the eigenvalue: an upper bound of :math:`\omega_\mathrm{max}^2`, as the
element eigenvalue inequality, Gershgorin's or Collatz-Wielandt's give it. An *estimate* is on the side of time: a
part's lower bound of its critical time step, made from such a bound
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepEstimate`). The *time increment* is what a step
takes, at most ``courant-number`` times the critical time step. The code, its options and its journal use these words
in this sense only.

How the parts combine
---------------------

A model may hold finite elements and particles. Their degrees of freedom are disjoint -- ``NED`` refuses a node that
both would carry -- so the matrix :math:`\boldsymbol{M}^{-1}\boldsymbol{K}` of the model is block diagonal in the two
parts, its eigenvalues are those of the parts, and the critical time step of the model is the **smallest** of the
parts' (as far as the parts' own stiffness goes; what penalty constraints add is explained below):

.. math::

   \Delta t = \text{courant-number} \times \min\left(\Delta t_\mathrm{crit}^\mathrm{elements},\,
   \Delta t_\mathrm{crit}^\mathrm{particles}\right) .

``courant-number`` is the one safety factor: the fraction of the estimated limit the analysis runs at. A part gives
its estimate as an object (:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepEstimate`), whose value
is the part's critical time step, and which, in a derived class, also holds what it was made from: what is added to a
part's stiffness can be added to its estimate there
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepEstimateOfTheDofs`). With penalty constraints, a
third value joins the minimum (below). It is not the constraints' own: it is the critical time step of the parts *with*
the constraints' rows added at the degrees of freedom they couple. It is never above the parts' own estimates, so the
minimum with them changes nothing while they are current; they are kept in it for the increments in which a part was
estimated after the constraints.

The particles' estimate has one factor of its own, below one, which is not a safety factor but part of what it
estimates: their stabilization (NSNI) acts on the increment, so their explicit scheme is not exactly the central
difference method with their tangent, and it was measured to be stable up to a fraction of the tangent's limit
(EdelweissMeshfree, ``STABLE_FRACTION_OF_THE_TANGENT_LIMIT``).

Penalty constraints
-------------------

A penalty constraint -- a contact, a support held by a penalty, a spring between two nodes -- is a set of springs. A
point :math:`p` of it measures a gap :math:`g_p`; the vector with which its spring acts on the degrees of freedom is the
gradient of the gap, :math:`\boldsymbol{a}_p = \partial g_p / \partial \boldsymbol{u}`, and a spring of stiffness
:math:`k_p` adds :math:`k_p\,\boldsymbol{a}_p\boldsymbol{a}_p^T` to the stiffness of the nodes it couples. For a contact
point, :math:`\boldsymbol{a}_p = [c, -N^m] \otimes \boldsymbol{n}` (the interpolation vector of the point on the slave
side, the shape functions of the master face, the normal), and :math:`k_p = \mathrm{d}f_n/\mathrm{d}g` is what the
contact law gives the point at its gap: the penalty for the linear law (times the area the point stands for, where the
contact is integrated over a surface), zero while the gap is open (:mod:`~edelweissfe.constraints.base.penaltylaw`).

Such a constraint owns no degrees of freedom: it couples those of the parts, and has no critical time step of its own
to take the minimum with -- the spring *adds* to the frequencies of the nodes it couples. A mass :math:`m` on a spring
:math:`k_1` (an element) and held by a penalty :math:`k_2` oscillates with :math:`\omega^2 = (k_1 + k_2)/m`, not with
the larger of the two. Adding the constraint's largest frequency to the parts' largest one would be safe but blunt: a
stiff contact far from the smallest element would lower the time increment of the whole model.

**The estimate: the constraint adds where it acts.** For every degree of freedom :math:`i` of the finite elements, let
:math:`\omega_i^2` be the largest :math:`\lambda_e = (2/\Delta t_e)^2` of the elements at :math:`i`, and let

.. math::

   r_i = \frac{1}{\sqrt{M_i}} \sum_j |K^\mathrm{c}_{ij}| \frac{1}{\sqrt{M_j}}

be the row sum of the absolute value of the constraints' stiffness, scaled by the lumped mass on both sides (every
constraint gives it: ``addBoundOfAbsoluteStiffnessTimes``). Then

.. math::

   \omega_\mathrm{max}^2 \le \Lambda = \max_i \left(\omega_i^2 + r_i\right) .

At a degree of freedom no constraint couples, :math:`r_i = 0` and the elements' estimate stands; a constraint lowers the
critical time step only if the frequency it adds at its nodes, on top of the elements' there, exceeds the largest one
of the model. For a mass :math:`m` held by a penalty :math:`k` alone, :math:`r = k/m` and the estimate is the exact
:math:`2\sqrt{m/k}`; for two equal masses joined by a penalty, :math:`r = 2k/m`, again exact.

**Why it is safe: lend the constraint a part of the mass.** The argument of the element eigenvalue inequality holds for
*any* split of the lumped mass among the pieces of the stiffness: if :math:`\boldsymbol{K} = \sum_a
\boldsymbol{K}_a` and :math:`\boldsymbol{M} = \sum_a \boldsymbol{M}_a` (all diagonal, nonnegative), then

.. math::

   \boldsymbol{x}^T\boldsymbol{K}\boldsymbol{x} = \sum_a \boldsymbol{x}^T\boldsymbol{K}_a\boldsymbol{x}
   \le \sum_a \lambda_a\, \boldsymbol{x}^T\boldsymbol{M}_a\boldsymbol{x}
   \le \max_a \lambda_a \; \boldsymbol{x}^T\boldsymbol{M}\boldsymbol{x} ,
   \qquad \lambda_a = \lambda_\mathrm{max}(\boldsymbol{M}_a^{-1}\boldsymbol{K}_a) .

The constraint has no mass of its own, so it is lent the fraction :math:`\theta_i = r_i/\Lambda` of the mass of every
degree of freedom it couples, and the elements keep the rest, :math:`(1 - \theta_i) M_i`.

* The constraint with the masses :math:`\theta_i M_i`: by the row sums with the weights :math:`\sqrt{\theta_i}`
  (Collatz-Wielandt), its largest eigenvalue is at most :math:`\max_i r_i/\theta_i = \Lambda`.
* An element with the masses :math:`(1-\theta_i)` times its own: its eigenvalue grows at most by :math:`1/(1 -
  \max_{i \in e}\theta_i)`, and :math:`\lambda_e \le \omega_i^2 \le \Lambda - r_i = \Lambda (1 - \theta_i)`
  at each of its nodes, so it stays below :math:`\Lambda`.

Every piece is below :math:`\Lambda` with its share of the mass, so the whole is. The argument is only as safe as each
:math:`\lambda_e` is a bound of its element: the estimate is each element's own, and the one of the Python elements
lies above their exact critical time step (EdelweissFE#323). The argument is the Rayleigh quotient's, which needs a
symmetric stiffness: the penalty springs are (a closed point with a frozen projection), and the elements' estimate
bounds their elastic stiffness.

**The particles: the rows inside their field matrix.** The tangent of gradient-enhanced particles is not symmetric, and
their estimate is not a bound per degree of freedom: it is the Perron root of their field matrix :math:`\boldsymbol{Q}`
of the largest weighted row sums per block of fields (EdelweissMeshfree,
``CriticalTimeStepEstimateFromTheFieldMatrix``), which bounds every eigenvalue of any matrix whose absolute value it
bounds block by block. So the constraint's rows go there. Entry by entry, :math:`|\boldsymbol{S} + \boldsymbol{R}| \le
|\boldsymbol{S}| + |\boldsymbol{R}|`, with :math:`\boldsymbol{S}` and :math:`\boldsymbol{R}` the mass-scaled stiffness
of the particles and of the constraint. A constraint that couples degrees of freedom of one field block :math:`g` adds
at most :math:`r_i` to the weighted row sum of :math:`i` within :math:`g`, where the weights are constant on that field;
so :math:`Q_{gg}` raised to :math:`\max_{i \in g} (Z_{ig} + r_i)`, with :math:`Z_{ig}` the particles' own row sum of
:math:`i` within :math:`g`, bounds the block row sums of :math:`|\boldsymbol{S}| + |\boldsymbol{R}|`, and the Perron
root of a nonnegative matrix grows with its entries. This holds for any matrix, symmetric or not. With carried weights,
the particles keep their weights constant per field in every field a constraint acts on, so that the rows add as they
are.

A constraint that couples particles and finite elements at once -- a projectile of elements on a slab of particles --
is taken with the larger of the two: :math:`\Lambda` of the elements, and the particles' field matrix with the rows
added. This is not proven: the elements' argument needs the Rayleigh quotient, which the unsymmetric particles do not
have, and the particles' needs a bound per block, which the elements do not give.

**With multi-point constraints** (a ``tie`` of a fine mesh to a coarse one, hanging nodes), a slave moves with its
masters, :math:`u_s = \sum_a N_{sa} u_a`, and the system that oscillates is the condensed one,
:math:`\boldsymbol{T}^T \boldsymbol{K} \boldsymbol{T}` over the independent degrees of freedom :math:`\boldsymbol{y}`,
:math:`\boldsymbol{u} = \boldsymbol{T}\boldsymbol{y}`, with the slave masses folded onto the masters, :math:`\tilde{M}_m
= M_m + \sum_s N_{sm} M_s`. A spring at a slave acts on its masters: its row sums are taken with the weights of the
masters, :math:`w_s = \sum_a |N_{sa}| w_a`, and folded onto them by :math:`|\boldsymbol{T}|`, since
:math:`|\boldsymbol{T}^T \boldsymbol{K}^\mathrm{c} \boldsymbol{T}| \le |\boldsymbol{T}|^T |\boldsymbol{K}^\mathrm{c}|
|\boldsymbol{T}|` entry by entry, whatever the signs of the coefficients.

So does an element at a slave: it sees :math:`u_s^2`, which the masters must pay for with mass. With coefficients that
are nonnegative and sum to one -- a ``tie``, the hanging nodes of linear elements -- Jensen's inequality,
:math:`(\sum_a N_{sa} u_a)^2 \le \sum_a N_{sa} u_a^2`, asks master :math:`m` for :math:`N_{sm} M_s`, just what the fold
gives it. The hanging nodes of a HEX20 or QUAD8 mesh are tied with the serendipity functions of the coarse face, which
are negative at some masters (:math:`-1/8` for a node at a quarter of a coarse edge, :math:`-1/4` at the corners for
one at the centre of a coarse face). Jensen's inequality fails then, and the fold gives a master less than the elements
at its slaves need; a negative coefficient even takes mass away. Cauchy-Schwarz still holds: with coefficients that
sum to one, their absolute values sum to :math:`\sigma_s = 1 + 2 n_s`, :math:`n_s` the sum of the absolute values of
the negative ones, and :math:`u_s^2 \le \sigma_s \sum_a |N_{sa}| u_a^2`. So

.. math::

   \boldsymbol{u}^T\boldsymbol{K}\boldsymbol{u} \le \sum_e \lambda_e\, \boldsymbol{u}_e^T \boldsymbol{M}_e
   \boldsymbol{u}_e \le \sum_i \omega_i^2 M_i u_i^2 \le \sum_m \hat{\omega}_m^2 \left(M_m + \sum_s \sigma_s
   |N_{sm}| M_s\right) y_m^2 = \sum_m \kappa_m \hat{\omega}_m^2\, \tilde{M}_m y_m^2 ,

with :math:`\hat{\omega}_m^2` the largest :math:`\lambda_e` of the elements at master :math:`m` *and at its slaves*
(:meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.foldLargestOntoMasters`), and the
ratio of the mass the elements need to the mass folded

.. math::

   \kappa_m = \frac{M_m + \sum_s \sigma_s |N_{sm}| M_s}{\tilde{M}_m}
   = 1 + \frac{1}{\tilde{M}_m} \sum_s 2 \left(N^-_{sm} + n_s |N_{sm}|\right) M_s \ge 1

(:meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.ratioOfTheMassLentToTheMassFolded`;
:math:`N^-` the absolute value of a negative coefficient). The bound of the master is :math:`\omega_m^2 = \kappa_m
\hat{\omega}_m^2`, and the argument above holds for the condensed system: the elements' own critical time step is
:math:`2/\sqrt{\max_m \omega_m^2}`, the penalty constraints add their folded rows to it. :math:`\kappa_m` is exactly one
where every slave of the master has nonnegative coefficients, so a ``tie`` and the hanging nodes of linear elements keep
the smallest critical time step of the elements; at the masters of HEX20 hanging nodes it is above one, the more so the
smaller the folded mass of a master is against what its slaves need. There the bound is safe but blunt: the folded
masses of such a mesh are lighter than the consistent condensed mass
:math:`\boldsymbol{T}^T\boldsymbol{M}\boldsymbol{T}` in some motions, and no bound made from the elements' own estimates
alone can see whether those motions are stiff. Without the fold of the largest, a stiff element at a slave would be
missing at its master just where a penalty there adds the most.

What the bound gives away is the Gershgorin step: the row sum is the exact frequency for springs between equal masses
and above it otherwise, and a normal oblique to the axes spreads the spring over the components of a node.

**Points that could close.** By default, the stiffness the constraints add enters from the state where the next
increment starts -- after the contact search and the refinement, when they are due, so that a point paired anew and the
smaller masses of a refinement are in it --: a point that closes *within* an increment adds its stiffness one increment
late. An increment above the limit amplifies an oscillation by a bounded factor, and the time increment is never raised
again within the step, so this is not an instability; the energy balance of ``NED`` (the kinetic energy may not exceed
the energy at the start of the step plus the external work) remains the check. With ``follow-critical-time-step``, the
time increment is raised again as soon as the critical time step allows it -- when a contact has opened, for one --,
and a point that closes one increment late would be met by the full raised increment. The constraints are then bounded
over the next increment as well. The leapfrog takes an increment of length :math:`s` as

.. math::

   \Delta \boldsymbol{u}(s) = s\, \boldsymbol{v} + \tfrac{1}{2} s (\Delta t_\mathrm{last} + s)\, \boldsymbol{a} ,

with the velocity :math:`\boldsymbol{v}` of the last increment, its length, and the acceleration :math:`\boldsymbol{a}`
at its end; :math:`s` is at most :math:`\Delta t_\mathrm{max}`, the largest time increment the critical time step with
the constraints as they are now allows -- ``courant-number`` times it --, at most ``maxInc``: the constraints bounded
over the increment can only lower it.
With the pairing and the normals held, as within an increment, a gap is affine in the displacements, and changes by
:math:`s\, \dot{g}_v + \tfrac{1}{2} s (\Delta t_\mathrm{last} + s)\, \dot{g}_a`; each part, where it closes the gap,
closes it most at :math:`\Delta t_\mathrm{max}`. So within the increment, whatever its length, the gap is at least

.. math::

   g + \min(0, \Delta g_v) + \min(0, \Delta g_a) ,

with the changes :math:`\Delta g_v, \Delta g_a` of the gap by the two parts at :math:`\Delta t_\mathrm{max}`. A
contact that bounds its springs gives them this smallest gap, with the gradient of now
(:meth:`~edelweissfe.constraints.base.constraintbase.NextIncrement.smallestGaps`), and sums over its springs: a point
that could close within the next increment has its spring in the estimate before it closes, and a law whose stiffness
grows with the depth is stiffest there. A constraint bounded from its tangent is evaluated in the states that bound the
increment -- now, and moved on by either part or by both
(:meth:`~edelweissfe.constraints.base.constraintbase.NextIncrement.statesWithin`) --, and takes the largest absolute
value of each entry. That is a bound where each spring that can switch acts on degrees of freedom of its own in the
constraint's block: each spring is closed deepest in a state of its own, and the solver adds the row sums of the block
to the rows of the model only afterwards, where the springs that meet at a node are summed. The contacts with a
deformable master are of that kind: ``nodeToDeformableSurfacePenalty`` gives each slave node, and
``surfaceToDeformableSurfacePenalty`` each contact point, a copy of its own of the nodes it couples -- of its master
facet, and of its slave face -- in the block. Where springs that close in different states share a degree of freedom
within the block, they are never in one tangent together, and the entry misses their sum: at the reference point of a
rigid body, on which every spring of a contact with a discrete rigid body acts, and, in
``surfaceToDiscreteRigidBodyPenalty``, at the nodes of a slave facet, which its contact points share. Every contact of
EdelweissFE is bounded from its tangent; a constraint that needs the bound over the next increment there gives its
springs. Taking the velocity
alone would not do: near the critical time step of the open state, the acceleration can turn a receding point around
within one increment. The prediction is exact for the leapfrog without damping; damping only slows a point down.
Raising and lowering the increment is not free: each change of the increment changes the energy the central difference
holds in a mode by a fraction of the order of :math:`(\omega \Delta t)^2`, small for the motion that carries the
energy, which lies far below the highest frequencies of a mesh, but not for a mode near the critical time step that
carries much of it; the energy balance remains the check.

**From the tangent, or from the springs.** By default, a constraint is bounded from the tangent it gives at the state
(:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`), formed for
this before every increment. A constraint whose tangent is costly, or that has none -- a contact evaluated without a
tangent in an explicit analysis --, gives its stiffness as springs instead: a spring :math:`p` with the vector
:math:`\boldsymbol{a}_p` with which it acts and the stiffness :math:`k_p` of its law -- the normal spring of a closed
point, a support's spring along the axis it holds. The row sums are then, with the weights :math:`\boldsymbol{w}` of
the columns, :math:`b_i \mathrel{+}= \sum_p |k_p|\, |a_{p,i}|\, (|\boldsymbol{a}_p| \cdot \boldsymbol{w})`
(:mod:`edelweissfe.constraints.base.penaltysprings`), one formula for every such constraint, and the constraint takes
the springs for its bound from the same code it takes them for its force, so that a change of the force cannot leave
the bound behind.

What the estimate does not see
------------------------------

The parts' estimates cover the stiffness of the parts themselves, and the penalty constraints add theirs where they act
(above), within a part and between the parts. Not seen are:

* **A point that closes within an increment.** By default, it enters the estimate one increment late; with
  ``follow-critical-time-step``, before it closes, for a constraint that bounds its springs, and for one bounded from
  its tangent whose springs that can switch act on degrees of freedom of their own in its block -- a contact with a
  deformable master (see "Points that could close").
* **With follow-critical-time-step, a constraint bounded from its tangent whose springs that close in different states
  share a degree of freedom within its block** -- a contact with a discrete rigid body, at its reference point, and
  ``surfaceToDiscreteRigidBodyPenalty`` at the nodes of a slave facet as well: over the next increment, it is seen
  only as far as one state closes its springs together.
* **The geometric stiffness of contact.** That of a contact with finite sliding, or of a rigid body whose normals rotate
  with it (of the order of the normal force over the size of a facet or the lever arm, small against the penalty), and
  the coupling of the normal and the tangential force of a sliding frictional contact (the friction coefficient times
  the normal stiffness; unsymmetric) -- for a constraint that bounds its springs; one bounded from its tangent has in
  its bound what its tangent holds.
* **A constraint that couples particles and finite elements at once**: taken with the larger of the two arguments,
  which is not proven (see "The particles").
* **A multi-point constraint whose coefficients do not sum to one** (a linearized rigid body): :math:`\kappa` assumes
  coefficients that sum to one.
* **A penalty on a field integrated by forward Euler** (a field with a damping and no inertia): its rows are added to
  no estimate.
* **A constraint that bounds nothing**: EdelweissMeshfree's ``ParticlePenaltyFacetContactExplicit`` gives neither a
  tangent nor its springs; its stiffness stays outside the estimate.
* **The tangent of a rate-dependent material.** The stiffness the particles are estimated with is that of a zero
  increment over a unit time; for a material with a viscosity, e.g. Duvaut-Lions, it is relaxed over that time, softer than the
  response within an explicit increment, and the estimate may be too large.
* **The particles' stabilization.** The factor of the particles' NSNI above was measured, on blocks of particles, not
  derived; a particle arrangement far from those may need a smaller one.

How to check: halve ``courant-number`` and compare the results, and watch the energy balance of ``NED``: kinetic energy
that appears from nowhere is the sign of a time increment above the true limit.

When an estimate is taken anew
------------------------------

The true critical time step changes as the analysis runs: elements are refined, particles move and deform, materials
stiffen and soften. Estimating in every increment would always be right, but an estimate of all particles costs a
noticeable part of an increment. So each part comes with a *policy*, decided by what its estimate depends on.

**The finite elements: once per mesh, and at the start of every step**
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfTheElements`). Their estimate depends on the
mesh, on the lumped mass, and on the wave speed of the material at the state of the estimate. An estimate of all
elements costs a full element pass, so the policy *assumes* that within a step neither the geometry nor the wave speed
of an element changes enough to use up the margin the courant number leaves. A softening material only raises the
true limit; the assumption fails where elements are compressed far at finite strain, which shrinks their geometry, or
where their material stiffens with the deformation: a hyperelastic solid in compression, a compacting material. There,
take a lower ``courant-number``, or shorter steps, since every step estimates anew. The elements are estimated anew
after a refinement, and at the start of every step, since a material property may have changed with the step
(``changeMaterialProperty``).

**The particles: when the margin may be used up**
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfEntitiesWithOwnInertia`). Their estimate depends
on their state: on their lumped masses, which change as they move over the nodes of the kernel functions, and on their
stiffness, which changes with the material and with the deformation of the particles. After an estimate :math:`v`,
``NED`` keeps in every increment the critical time step at worst, :math:`L`: a lower bound of the particles' critical
time step now, :math:`n` increments later,

.. math::

   L = v \, \min\left(1, \sqrt{\alpha}, \beta\right) (1 - \delta)^n ,

and takes a new estimate when :math:`\text{courant-number} \times L` falls below the time increment in use.

* :math:`\alpha = \min_i M_i / M_i^0 \le 1` is the smallest ratio of a lumped mass now to the same mass at the
  estimate (1 if none fell), :math:`\beta \le 1` the same for the damping of a forward-Euler field. This factor is
  **exact**. The estimate scales the stiffness with :math:`\boldsymbol{M}^{-1/2}` on both sides; if every mass is at
  least :math:`\alpha` times what it was, every scaled row sum, and with them the bound of
  :math:`\omega_\mathrm{max}^2`, is at most :math:`1/\alpha` times larger (the largest eigenvalue of a nonnegative
  matrix grows with its entries), so the estimate is at least :math:`\sqrt{\alpha}` times what it was. The damping
  enters linearly instead: the forward Euler limit :math:`2/\lambda_\mathrm{max}(\boldsymbol{C}^{-1}\boldsymbol{K})`
  is linear in :math:`\boldsymbol{C}`, while the central differences' :math:`2/\omega_\mathrm{max}` goes with the
  square root of :math:`\boldsymbol{M}`. The masses cost nothing to watch: ``NED`` assembles them in every
  increment anyway. A node that gets a mass for the first time (a particle reaches a kernel function's node, whose mass
  grows from zero) is a new row whose scaling has no bound: then the particles are estimated anew at once.
* :math:`\delta`, the option ``critical-time-step-decrease-per-increment``, is the largest relative fall per increment
  the stiffness may cause. This factor is an **assumption**. No theorem bounds it: the tangent of a material can change
  abruptly from one increment to the next. What makes it reasonable is what an increment is: a fraction of the time a
  wave needs to cross a particle, so in one increment the configuration moves by a fraction of a particle, and the
  stiffness, which follows the configuration, changes by a fraction of itself. Its default was measured: in the
  impact of a steel projectile on a concrete slab of gradient-enhanced particles, the fall of the
  particles' estimate not explained by their masses stayed below half the default in every single increment, and far
  below it over several increments. There, the fall came from particles at the rim of the crater that are stretched
  far: the nonlocal field of a gradient-enhanced particle is balanced with gradients in the reference configuration,
  which grow with the stretch, while its micro-inertia does not -- a smooth change from increment to increment, which
  only an estimate sees, not the masses. A larger :math:`\delta` is more cautious and estimates more often; 1 estimates
  in every increment.

Where penalty constraints couple the particles, the same critical time step at worst is kept for the particles' rows
with the constraints added: the particles' stiffness may have grown by :math:`1/f^2`, :math:`f = \min(1,
\sqrt{\alpha}, \beta)(1 - \delta)^n`, the constraints' rows are those of now. The particles are estimated anew also
when ``courant-number`` times that one falls below the time increment in use -- the margin a contact at the particles'
nodes leaves is smaller than their own.

What the policy does in practice follows from the margin. Where the particles set the time increment, the increment in
use *is* courant-number times their last estimate, the margin is used up after one increment, and they are estimated in
every increment -- as they must be, since their estimate is what limits the analysis. Where another part sets it, the
estimate of the particles is above the increment in use by some factor :math:`m`, and it is taken anew only every
:math:`\ln m / -\ln(1-\delta)` increments, unless their masses fall faster than expected.

**The penalty constraints: before every increment**
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepWithThePenaltyConstraints`). Their stiffness follows
the state of the contact -- which points are closed, and how deep, for a law whose stiffness grows with the
penetration --, and changes from increment to increment; their row sums cost about as much as their force, so they are
formed before every increment, from the state it starts from, on the mesh and the connectivity it uses (after a
refinement or a contact search, when one is due), and added to the parts' last estimates. Only this addition is
renewed; the parts keep their own policies.

The code
--------

.. automodule:: edelweissfe.solvers.base.criticaltimestep
   :members:

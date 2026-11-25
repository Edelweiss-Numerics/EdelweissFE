Theory: the critical time step of an explicit analysis
======================================================

An explicit analysis (``NED``, :doc:`/analysis/solvers`) solves no system of equations: it divides the net force by a
lumped mass and steps on. The price is that it is stable only below a time increment, the *critical time step*. This
page explains where that limit comes from, why EdelweissFE estimates it instead of computing it, how the estimates of
the parts of a model are combined, and when each estimate is taken anew. The code is
:mod:`~edelweissfe.solvers.base.criticaltimestep`.

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
  Marmot solves it with the element's own strain operator and the largest wave speed of its material, which holds for
  every element order and distortion -- where the classic "element length over wave speed" is exact only for a linear
  bar.
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

How the parts combine
---------------------

A model may hold finite elements and particles. Their degrees of freedom are disjoint -- ``NED`` refuses a node that
both would carry -- so the matrix :math:`\boldsymbol{M}^{-1}\boldsymbol{K}` of the model is block diagonal in the two
parts, its eigenvalues are those of the parts, and the critical time step of the model is the **smallest** of the
parts':

.. math::

   \Delta t = \text{courant-number} \times \min\left(\Delta t_\mathrm{crit}^\mathrm{elements},\,
   \Delta t_\mathrm{crit}^\mathrm{particles}\right) .

``courant-number`` is the one safety factor: the fraction of the estimated limit the analysis runs at.

The particles' estimate has one factor of its own, below one, which is not a safety factor but part of what it
estimates: their stabilization (NSNI) acts on the increment, so their explicit scheme is not exactly the central
difference method with their tangent, and it was measured to be stable up to a fraction of the tangent's limit
(EdelweissMeshfree, ``STABLE_FRACTION_OF_THE_TANGENT_LIMIT``).

When an estimate is taken anew
------------------------------

The true critical time step changes as the analysis runs: elements are refined, particles move and deform, materials
stiffen and soften. Estimating in every increment would always be right, but an estimate of all particles costs a
noticeable part of an increment. So each part comes with a *policy*, decided by what its estimate depends on.

**The finite elements: once per mesh**
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfTheElements`). Their estimate uses the elastic
wave speed of the material, not its state. A softening material only raises the true limit, so the estimate stays safe
until the mesh changes; it is taken anew after a refinement and at the start of every step.

**The particles: when the margin may be used up**
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfEntitiesWithOwnInertia`). Their estimate depends
on their state: on their lumped masses, which change as they move over the nodes of the kernel functions, and on their
stiffness, which changes with the material and with the deformation of the particles. After an estimate :math:`v`,
``NED`` keeps in every increment a lower bound of what the estimate can be now, :math:`n` increments later:

.. math::

   L = v \, \min\left(1, \sqrt{\alpha}, \beta\right) (1 - \delta)^n ,

and takes a new estimate when :math:`\text{courant-number} \times L` falls below the time increment in use.

* :math:`\alpha` is the largest relative fall of a lumped mass since the estimate, :math:`\beta` the same for the
  damping of a forward-Euler field. This factor is **exact**. The estimate scales the stiffness with
  :math:`\boldsymbol{M}^{-1/2}` on both sides; if every mass is at least :math:`\alpha` times what it was, every scaled
  row sum, and with them the bound of :math:`\omega_\mathrm{max}^2`, is at most :math:`1/\alpha` times larger (the
  largest eigenvalue of a nonnegative matrix grows with its entries), so the estimate is at least :math:`\sqrt{\alpha}`
  times what it was. The masses cost nothing to watch: ``NED`` assembles them in every
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

What the policy does in practice follows from the margin. Where the particles set the time increment, the increment in
use *is* courant-number times their last estimate, the margin is used up after one increment, and they are estimated in
every increment -- as they must be, since their estimate is what limits the analysis. Where another part sets it, the
estimate of the particles is above the increment in use by some factor :math:`m`, and it is taken anew only every
:math:`\ln m / -\ln(1-\delta)` increments, unless their masses fall faster than expected.

The code
--------

.. automodule:: edelweissfe.solvers.base.criticaltimestep
   :members:

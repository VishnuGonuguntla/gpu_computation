# Molecular dynamics and the Discrete Element Method

This document covers the physics behind three related exercises that build on each other:

- `2_molecularDyn/` — brute-force O(N²) Lennard-Jones molecular dynamics.
- `3_molecularDynOpt/` — the same simulation, sped up with a cell-list neighbor search.
- `4_DEM/` — the Discrete Element Method: rigid, frictional, granular spheres instead of point particles.

All three share the same simulation skeleton: initialize particles on a lattice, compute forces, integrate with Velocity-Verlet, repeat.

## 1. Initialization

Particles are placed on a regular cubic lattice rather than randomly, so that no two particles start overlapping (`kernelInitSolver` in `2_molecularDyn/src/kernels.cuh` and `4_DEM/src/kernels.cuh`):

```
gridSize = ceil(cbrt(nParticles))
pos[n]   = ((i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing)
```

Initial velocities are drawn from a normal distribution with `sigma = sqrt(kT / mass)` — a Maxwell-Boltzmann-like approximation for a gas/liquid at temperature `T`, giving the ensemble a physically reasonable starting kinetic energy spread instead of starting every particle at rest.

## 2. The Lennard-Jones potential (`2_molecularDyn`, `3_molecularDynOpt`)

Pairwise interactions between particles use the **Lennard-Jones 12-6 potential**, a standard model for how two neutral atoms/molecules attract at long range (van der Waals attraction, the `-r^-6` term) and repel sharply at short range once their electron clouds overlap (the `r^-12` term):

```
U(r) = 4*eps * [ (sigma/r)^12 - (sigma/r)^6 ]
```

`eps` sets the depth of the attractive well (how strongly particles stick together), `sigma` sets the characteristic distance at which the potential crosses zero. The force is the negative gradient of this potential; the code computes it directly on the squared distance to avoid a `sqrt`, and uses repeated squaring to avoid `pow()` (`kernelComputeForceLJ`):

```
sr2  = (sigma/r)^2            // computed as sigma^2 / dist2, dist2 = r^2
sr6  = sr2^3
sr12 = sr6^2
F/r  = 24*eps * (2*sr12 - sr6) / dist2      // force magnitude divided by r, folded into the per-axis component below
F_x += (F/r) * (x_i - x_j)                   // and similarly for F_y, F_z
```

Interactions are truncated at a cutoff radius (`rCutoff`) — Lennard-Jones forces decay fast enough that particles farther apart than a few `sigma` contribute negligibly, so skipping them is a safe and large performance win. `2_molecularDyn` finds neighbors by checking every other particle (O(N²) per step); `3_molecularDynOpt` replaces that with the cell list described below.

Energy diagnostics (`kernelCalculateEnergyPBC`) track kinetic energy `KE = 0.5*m*v^2` and the LJ potential energy per pair, summed with atomic adds across threads — useful for verifying the simulation is physically sane (e.g. total energy should stay roughly constant in an isolated system).

## 3. Velocity-Verlet integration

All three projects integrate with **Velocity-Verlet**, split across two kernel launches per timestep so that forces at the *new* positions can be computed in between (`kernelFirstIntegratePBC`, force kernel, `kernelFinalIntegratePBC`):

```
# kernelFirstIntegratePBC — position update + first half of the velocity update
x(t+dt)      = x(t) + v(t)*dt + 0.5*a(t)*dt^2
v(t+dt/2)    = v(t) + 0.5*a(t)*dt

# (forces/accelerations recomputed at the new positions x(t+dt))

# kernelFinalIntegratePBC — second half of the velocity update
v(t+dt)      = v(t+dt/2) + 0.5*a(t+dt)*dt
```

Velocity-Verlet is preferred over plain forward Euler for this kind of simulation because it is **symplectic** — it conserves energy much better over long integrations (thousands of timesteps), which matters here since these simulations run far longer than the short MPPI rollouts described in [`docs/mppi-physics.md`](mppi-physics.md). It is also time-reversible and only needs one force evaluation per step (done at the new position, straddling the two half-velocity updates above).

## 4. Cell-list neighbor search (`3_molecularDynOpt`)

Checking every pair of particles is O(N²) and wasteful once the interaction cutoff is much smaller than the simulation box — most pairs are far enough apart to contribute zero force. The cell list (`CellList.cpp`/`.hpp`, and `kernelBuildCellList` in the DEM kernels) fixes this by partitioning the box into a grid of cells at least as large as the cutoff radius:

```
numCellsPerDim = floor(boxSize / cutoff)
cellSize       = boxSize / numCellsPerDim
cellIndex(particle) = floor(pos / cellSize), wrapped modulo numCellsPerDim per axis (periodic boundaries)
```

Each cell stores the index of one particle (`cell[]`), and every particle stores the index of the *next* particle in the same cell (`cellIndex[]`) — a classic head/next singly-linked list per cell, rebuilt every step as particles move. Because the cutoff radius can never reach past a cell's immediate neighbors (cells are sized ≥ cutoff), a particle only needs to scan its own cell plus its 26 neighboring cells (the DEM kernel's `dx,dy,dz ∈ {-1,0,1}` triple loop) instead of the whole system — turning the force computation into roughly O(N) work instead of O(N²), at the cost of periodically rebuilding the list as particles move between cells.

## 5. Discrete Element Method (`4_DEM`)

`4_DEM` replaces point particles with **rigid spheres** that have a radius, an angular velocity, and an orientation quaternion, and replaces the smooth Lennard-Jones potential with **contact-only** forces: two spheres interact only while they physically overlap (`dist < r_i + r_j`), modeling granular materials like sand or ball bearings rather than a gas.

### Normal force: spring-damper

When two spheres overlap by `overlap = (r_i + r_j) - dist`, the normal contact force is a linear spring (pushing them apart, proportional to how deep the overlap is) plus a damper (dissipating energy, proportional to how fast they're approaching along the contact normal):

```
F_n = k * overlap - gamma * v_rel_normal
```

`k` is the contact stiffness (effectively how "hard" the particles are — stiffer means less overlap for the same force, but also a smaller stable timestep), and `gamma` is a damping coefficient that models inelastic collisions (without it, spheres would bounce forever).

### Tangential force: capped Coulomb friction

The relative *surface* velocity at the contact point — not just the centers' relative velocity, but including each sphere's own spin via `ω × r̂` — determines a sliding friction force, capped by the normal force through the Coulomb friction law (`kernelFrictionalForce`):

```
v_surface = v_rel - (v_rel · n̂)*n̂ + (r_i*ω_i + r_j*ω_j) × n̂     // relative velocity tangent to the contact, including spin
F_t = -v̂_surface * min(gamma * |v_surface|, mu * F_n)
```

This is a regularized Coulomb model: for small sliding speeds the tangential force grows like viscous damping (`gamma * v_surface`), but it's clamped so it can never exceed `mu * F_n` — a sphere can never be held in place by friction stronger than "friction coefficient × normal force," exactly like a block that starts sliding once pushed hard enough.

### Torque and rotation

The tangential force acts at the contact point, one radius away from the sphere's center, producing a torque `τ = -r * (n̂ × F_t)`. Angular acceleration follows from the moment of inertia of a solid sphere, `I = 0.4 * m * r²`:

```
alpha = tau / I
```

Orientation is tracked as a unit quaternion and integrated with the standard rigid-body quaternion kinematics equation `dq/dt = 0.5 * ω ⊗ q` (quaternion multiplication of the angular velocity, treated as a pure quaternion, with the current orientation), followed by renormalizing the quaternion each step to counteract numerical drift away from unit length.

### Gravity and boundaries

Gravity is added as a constant `-g` acceleration. The simulation box walls behave like any other contact: `enforceBoundaries` applies the same spring-damper normal force when a sphere's center comes within one radius of a wall, and reflects the velocity component into the wall (`v = -v`) — a simple elastic/damped bounce rather than a true contact-mechanics wall model.

Both position and angular-velocity integration follow the same Velocity-Verlet split-step pattern described in §3, just applied to linear and angular quantities together.

## 6. Output and visualization

`2_molecularDyn`, `3_molecularDynOpt`, and `4_DEM` all write particle positions as ASCII VTK `UNSTRUCTURED_GRID` files (`writeVTK`), meant to be opened in [ParaView](https://www.paraview.org/) for 3D playback — there is no built-in renderer. This is a different (and independent) visualization path from `mppi`'s own `visualizer.py`, which does 2D matplotlib playback of racing telemetry rather than 3D particle data (see [`docs/mppi-physics.md`](mppi-physics.md)).

## References

- Source: `2_molecularDyn/src/kernels.cuh`, `2_molecularDyn/src/Solver.cpp`
- Source: `3_molecularDynOpt/src/CellList.cpp`, `3_molecularDynOpt/src/CellList.hpp`
- Source: `4_DEM/src/kernels.cuh`

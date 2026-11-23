# MPPI racing controller: vehicle model, costs, and the sampling algorithm

This document explains the physics and control theory behind `mppi/`, the capstone project: a multi-car racing controller that drives each car around a track using **Model Predictive Path Integral (MPPI)** control. The serial (`mppi/src/MPPI.cpp`) and CUDA (`mppi/src/kernels.cuh`) backends implement the exact same math — they share the header-only `dynamics.h` and `cost.h` so the two cannot drift apart.

## 1. Vehicle model: the single-track (bicycle) model

Each car is reduced to a **bicycle model**: one front wheel, one rear wheel, both on the car's centerline. State and parameters (`mppi/src/types.h`):

```
state:  (x, y, psi, vx, vy, r)
  x, y   — position in the world frame
  psi    — heading angle
  vx, vy — longitudinal/lateral velocity, in the car's own body frame
  r      — yaw rate (dpsi/dt)

params: (M, I_z, a, b, C_f, C_r, mu)
  M      — car mass
  I_z    — yaw moment of inertia
  a, b   — distance from the center of gravity to the front/rear axle
  C_f, C_r — front/rear cornering stiffness
  mu     — tire-road friction coefficient
```

Working in the body frame (`vx`, `vy`, `r`) rather than world-frame velocity is what makes the tire force model below tractable — tire forces are naturally expressed relative to the direction the wheel is rolling, not the direction the car is pointing on the map.

### Slip angle

The **slip angle** α is the angle between where a tire is pointed and the direction it is actually traveling. It is the single quantity that determines how much lateral grip a tire has lost. For the front and rear axle (`dynamics::slip_angle`):

```
alpha_front = atan((vy + a*r) / vx) - steering
alpha_rear  = atan((vy - b*r) / vx)
```

`a*r` and `b*r` are the extra lateral velocity each axle experiences purely from the car rotating about its center of gravity (yawing). `vx` is clamped away from zero (`±0.01`) to avoid dividing by zero at a standstill.

### Brush-tire lateral force

Real tires don't produce force linearly forever — grip saturates as slip angle grows, and the tire eventually just slides. The **brush-tire model** (`dynamics::brush_force`) captures this with a cubic polynomial near zero slip that smoothly transitions into a constant, fully-saturated sliding force at large slip:

```
max_friction = mu * F_z
xi = sqrt(max_friction^2 - u_F^2) / max_friction      // fraction of grip budget left after longitudinal force u_F
gamma = atan(3*xi*mu*F_z / C)                           // slip angle at which the tire fully saturates

if |alpha| >= |gamma|:        // past saturation — sliding
    F_y = -mu * xi * F_z * sign(alpha)
else:                          // brush-tire cubic region
    F_y = -C*tan(alpha) + (C^2 / (3*xi*mu*F_z))*|tan(alpha)|*tan(alpha)
          - (C^3 / (27*xi^2*mu^2*F_z^2)) * tan(alpha)^3
```

- `F_z` is the tire's normal (vertical) load — more load means more available grip.
- `xi` implements a **friction ellipse**: the longitudinal force `u_F` (from the throttle) already "spends" some of the tire's total available friction `mu*F_z`, so there's less budget left over for lateral (cornering) force. This is why braking or accelerating hard while turning reduces how sharply a car can corner.
- The static normal load on each axle comes from the car's weight distributed by its geometry:
  ```
  F_zF = M*g*b / (a+b)
  F_zR = M*g*a / (a+b)
  ```

### Integration: forward Euler

Given the two tire forces, the car's state is advanced with a plain forward-Euler step (`dynamics::step`), matching the default integrator used by the MPPI-Generic reference (arXiv:2409.07563):

```
d_vx  = (u_F - F_yF*sin(steer)) / M + r*vy
d_vy  = (F_yF + F_yR) / M - r*vx
d_r   = (a*F_yF - b*F_yR) / I_z
d_x   = vx*cos(psi) - vy*sin(psi)
d_y   = vx*sin(psi) + vy*cos(psi)
d_psi = r

next_state = current_state + d_state * dt
```

The `r*vy`/`r*vx` cross-coupling terms are the usual rotating-frame (Coriolis-like) corrections that appear because `vx`/`vy` are expressed in the car's own rotating body frame, not a fixed world frame.

Forward Euler is a deliberate, simple choice here: MPPI rolls out hundreds of short trajectories per control step, so an integrator that's cheap to evaluate matters more than one that's long-term energy-accurate. Contrast this with the molecular-dynamics projects (see [`docs/molecular-dynamics.md`](molecular-dynamics.md)), which integrate for thousands of steps and need Velocity-Verlet's better energy behavior instead.

## 2. Cost function (`mppi/src/cost.h`)

Every rollout accumulates a scalar cost that MPPI will later use to decide which sampled control sequences were "good." Costs are evaluated at every simulated timestep (running cost) plus once at the end of the horizon (terminal cost).

- **Off-track penalty** (`position_cost`): projects the car's position onto the nearest segment of the track centerline polyline (closest-point-on-segment, looping back from the last waypoint to the first). If that distance exceeds `track_width/2 + margin`, a large fixed penalty (`cost_offtrack`) is applied — effectively a hard wall.
- **Obstacle penalty**: a large fixed penalty (`cost_obstacle`) if the car's position is within `radius + margin` of any static circular obstacle loaded from `trackData.txt`.
- **Inter-car collision penalty** (`collision_cost`): compares this car's rolled-out position at timestep `t` against every other car's *currently planned* path at the same `t`, penalizing (`cost_collision`) if they come within `collision_radius` of each other — this is what keeps multiple cars from planning straight through one another.
- **Control/speed costs**: quadratic penalties on steering, throttle, and the (target speed − actual speed) error, so the controller prefers smooth, moderate inputs that track a desired speed rather than jerking the wheel or speeding/slowing erratically.
- **Terminal progress reward**: `-cost_progress * distance(end, start)` — a *negative* cost (i.e. a reward) proportional to how far the car's horizon-end position is from its horizon-start position, which is what drives the car forward instead of sitting still (sitting still is otherwise the cheapest possible trajectory).

## 3. The MPPI algorithm

MPPI is a sampling-based model-predictive controller: instead of solving an optimization problem analytically, it simulates many random perturbations of a nominal control sequence, weighs each by how well it did, and blends them into an improved nominal sequence. Each physical control step (`MPPI::compute` in `mppi/src/MPPI.cpp`, or the kernel pipeline in `mppi/src/kernels.cuh`) does the following, for `M` samples over a horizon of `T` steps:

1. **Sample**: for every rollout `k` and every horizon step `t`, draw Gaussian noise `(ε_steer, ε_throttle) ~ N(0, Σ)` and add it to the current nominal control sequence, then clamp to actuator limits.
2. **Roll out**: simulate the bicycle model forward `T` steps with `dynamics::step` using the perturbed controls, accumulating the running cost at every step plus the terminal cost at the end. This gives a total cost `J_k` for rollout `k`.
3. **Baseline**: `rho = min_k J_k` — subtracting the best observed cost before exponentiating keeps the weights below from overflowing/underflowing.
4. **Importance weights**: `w_k ∝ exp(-(J_k - rho) / lambda)`. Low-cost rollouts get weight close to 1; high-cost rollouts get weight close to 0. `lambda` (the "temperature") controls how sharply the controller favors the very best samples versus averaging broadly over many decent ones.
5. **Update**: the nominal control sequence is updated by adding the weighted average of all the sampled noise sequences, `U_t ← u_t + Σ_k w_k * ε_t^k / Σ_k w_k`, then clamped.
6. **Apply and shift**: the car applies only the *first* control of the newly updated nominal sequence (receding-horizon / MPC pattern), then the whole nominal sequence is shifted left by one step and zero-padded at the end (`MPPI::shift`) — a warm start for the next control step, since the previous plan is still a decent guess for the shifted horizon.

### Why this maps well to a GPU

Every one of the `M` rollouts is independent of every other rollout — they share only the current state and the nominal sequence they perturb, and don't interact with each other during simulation (inter-car collision cost only reads *other cars'* already-fixed predicted paths from the *previous* step, not other samples of the *same* car). That makes the inner loop over samples an embarrassingly parallel Monte-Carlo problem: `mppi/src/kernels.cuh` assigns one CUDA thread per `(car, sample)` pair for the rollout kernel, uses `curand` for per-thread Gaussian noise generation, and runs a parallel weighted-reduction kernel to combine the `M` noise samples into the updated nominal trajectory per car per timestep — the same five steps above, just spread across thousands of threads instead of a serial `for` loop.

## References

- MPPI-Generic, Algorithm 1: [arXiv:2409.07563](https://arxiv.org/abs/2409.07563)
- Source: `mppi/src/types.h`, `mppi/src/dynamics.h`, `mppi/src/cost.h`, `mppi/src/MPPI.cpp`, `mppi/src/kernels.cuh`

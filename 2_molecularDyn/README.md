Mandatory Submission Guidelines

    Submit your works in groups. Each group should consist of three members, but groups of two or four members are also acceptable. You can use the StudOn forum to find team members or ask around in the computer exercise.
    The exam will cover detailed questions about the exercises and their contents. Please make sure you understood everything.
    Your solution has to include a suitable Makefile.
    To pass the exercise you need to
        upload your solution as a group via StudOn and
        show your solution to a tutor in the computer exercise.

Exercise 2, Part 1: Molecular Dynamics

Your next application is a particle simulator in the domain of molecular dynamics. Your implementation needs to fulfill the following conditions:

    Particles exist in a 3D space and all calculations are performed in 3D.
    No boundary conditions are imposed (i.e. the simulation domain is infinite).
    Initial conditions can be specified at execution time.
        Initial particle positions, velocities and masses are provided via input file (you are free to fix the initial particle accelerations or read them from file as well).
        Other relevant parameters (time step length, number of time steps, \sigma, \epsilon, etc.) and the particle input file name are either specified via command line arguments or via configuration file.
    Forces are calculated according to the Lennard-Jones-Potential (c.f. lecture slides).
    The numerical integration is done via the Velocity Verlet algortihm (c.f. lecture slides).
    All relevant calculations (force update and integration) are done on the GPU.
    Particle positions can be visualized over time.

For most teams, adding a visualization will correspond to implementing an offline rendering approach: the simulation writes all particles to file (using e.g. the VTK file format) at defined points of simulation time (e.g. every 100 time steps). Later, the emitted files are visualized with third-party software (e.g. Paraview). For your convenience, we provide an exemplary vtk output. After opening it in Paraview, glyphs can be added to visualize the particle data.

You are, however, free to also choose a different approach if you are familiar with other techniques such as rendering the particles directly while the simulation is running.

Your tasks are as follows:

    Implement your GPU-accelerated particle simulator fulfilling the restrictions detailed above.
    Check your implementation with simple test cases consisting of only two particles with varying distances:
        stable distance, i.e. particles don't move
        attraction distance
        repelling distance
    Set up at least three visually interesting test cases with a varying number of particles. Choose suitable initial conditions (i.e. positions, velocities and masses). One interesting scenarios could be the collision of two blocks of particles with varying masses.
    Compare the performance (mean execution time per time step) for your test cases and relate it to the respective number of particles.
    Post one representative visual result of your simulations in the StudOn group (either a screenshot of your visualization or a link to a short animation/ video).
    Upload your code, input files (or script generating the input files) and result image via StudOn.



My Solution:

In kernelCalculateEnergyPBC and kernelComputeForceLJ each for loop will work on one single particle. and since complexity is N^2, we have to go through another loop, so the index that we get is the particle that we should work on, then what does the thread number specify? (are both of them the same?)
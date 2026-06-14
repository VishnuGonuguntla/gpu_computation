Mandatory Submission Guidelines

    Submit your works in groups. Each group should consist of three members, but groups of two or four members are also acceptable. You can use the StudOn forum to find team members or ask around in the computer exercise.
    The exam will cover detailed questions about the exercises and their contents. Please make sure you understood everything.
    Your solution has to include a suitable Makefile.
    To pass the exercise you need to
        upload your solution as a group via StudOn and
        show your solution to a tutor in the computer exercise.

Exercise 3: Discrete Element Method

Your next task is modifying your existing molecular dynamics application to be able to simulation granular material. The following steps are meant as a guideline, but feel free to change their order if necessary:

    Particles are now rigid bodies instead of point masses. You may limit your implementation to spherical particles. Adapt your input files and data structures accordingly, i.e. each particle has an additional radius.
    Adapt you visualization to honor the radii of simulated particles.
    Update your force calculation:
        Particles only interact if they are overlapping. Implement a suitable contact detection.
        Calculate forces according to the spring-damper model (c.f. lecture).
    Adapt your acceleration structure as necessary.
    Add gravity. It needs to be configurable at execution time (parameter file or command line argument).
    Add fixed boundaries. The minimum requirement is that your domain is a box with walls on each side. More complex boundaries, either as domain boundaries or as boundaries embedded in the simulation domain (obstacles), can be used to set up more interesting show cases.
    Set up at least three visually interesting test cases with a varying number of particles.
    Example case for verification: Let a single particle fall under gravity, and plot kinetic energy decay over time
    Post one representative visual result of your simulations in the StudOn group (either a screenshot of your visualization or a link to a short animation/ video).
    Upload your code, input files (or script generating the input files) and result image via StudOn.Particle Simulations - Discrete Element Method

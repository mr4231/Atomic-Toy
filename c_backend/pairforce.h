#ifndef PAIRFORCE_H
#define PAIRFORCE_H

/* Compute the same 3D harmonic cutoff forces as src.physics.forces. */
void pair_forces_3d(
    int n_particles,
    const double *positions,
    double cutoff,
    double spring_constant,
    double equilibrium_distance,
    double *forces
);

#endif

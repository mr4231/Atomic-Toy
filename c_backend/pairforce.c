#include "pairforce.h"

#include <math.h>
#include <stddef.h>

void pair_forces_3d(
    int n_particles,
    const double *positions,
    double cutoff,
    double spring_constant,
    double equilibrium_distance,
    double *forces
) {
    int i, j, axis;
    const double cutoff_squared = cutoff * cutoff;

    /* Start from a zero net force before adding each unordered particle pair. */
    for (i = 0; i < n_particles * 3; ++i) {
        forces[i] = 0.0;
    }

    for (i = 0; i < n_particles; ++i) {
        for (j = i + 1; j < n_particles; ++j) {
            double displacement[3];
            double distance_squared = 0.0;
            double distance;
            double scale;

            for (axis = 0; axis < 3; ++axis) {
                displacement[axis] = positions[3 * i + axis] - positions[3 * j + axis];
                distance_squared += displacement[axis] * displacement[axis];
            }
            if (distance_squared <= 1e-24 || distance_squared > cutoff_squared) {
                continue;
            }

            distance = sqrt(distance_squared);
            scale = -spring_constant * (distance - equilibrium_distance) / distance;
            for (axis = 0; axis < 3; ++axis) {
                const double contribution = scale * displacement[axis];
                forces[3 * i + axis] += contribution;
                forces[3 * j + axis] -= contribution;
            }
        }
    }
}

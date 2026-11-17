#include "pairforce.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const double positions[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double forces[6];
    pair_forces_3d(2, positions, 2.0, 10.0, 0.8, forces);

    /* here r=1 and r0=0.8, particle 0 is pulled towards positive x. */
    if (fabs(forces[0] - 2.0) > 1e-12 || fabs(forces[3] + 2.0) > 1e-12) {
        fprintf(stderr, "pair-force reference test failed\n");
        return 1;
    }
    puts("C pair-force reference test passed");
    return 0;
}

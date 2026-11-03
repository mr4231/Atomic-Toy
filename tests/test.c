/* Optional C smoke test; compile with c_backend/pairforce.c when testing the C kernel. */
#include "../c_backend/pairforce.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const double positions[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double forces[6];
    pair_forces_3d(2, positions, 2.0, 10.0, 0.8, forces);

    if (fabs(forces[0] - 2.0) > 1e-12 || fabs(forces[3] + 2.0) > 1e-12) {
        fputs("C smoke test failed\n", stderr);
        return 1;
    }
    puts("C smoke test passed");
    return 0;
}

#include "common_utils.h"

real real_add(real lhs, real rhs) { return lhs + rhs; }
real real_sub(real lhs, real rhs) { return lhs - rhs; }
real real_mul(real lhs, real rhs) { return lhs * rhs; }
real real_div(real lhs, real rhs) { return lhs / rhs; }

/* lhs ^ rhs */
real real_pow(real lhs, real rhs) { return pow(lhs, rhs); }

/* rhs ^ lhs */
real real_pow_rev(real lhs, real rhs) { return pow(rhs, lhs); }

real real_abs(real x) { return (x>=0) ? x : -x; }

int sign(real x) {
    return (x >= 0) ? 1 : -1;
}

int isanum(char* s) {
    int i;
    for (i = 0; s[i] != 0; i++) {
        if (!(('0' <= s[i]) && (s[i] <= '9')))
            return 0;
    }
    return 1;
}
#include "common_utils.h"

real real_add(real lhs, real rhs) { return lhs + rhs; }
real real_sub(real lhs, real rhs) { return lhs - rhs; }
real real_mul(real lhs, real rhs) { return lhs * rhs; }
real real_div(real lhs, real rhs) { return lhs / rhs; }
real real_pow(real lhs, real rhs) { return pow(lhs, rhs); }

int isanum(char* s) {
    int i;
    for (i = 0; s[i] != 0; i++) {
        if (!(('0' <= s[i]) && (s[i] <= '9')))
            return 0;
    }
    return 1;
}
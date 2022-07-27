#!/bin/bash
# Script to compile and execute a c program

PATH_SRC="./source/c"
PATH_OUT="./output"

mkdir -p $PATH_OUT

gcc -ansi -Wall -Wextra -Werror -pedantic-errors \
    -D FLAG_DEBUG \
    -D FLAG_PRINTD \
    -D FLAG_ASSERTD \
    -I $PATH_SRC \
    $PATH_SRC/generics/common_utils.c \
    $PATH_SRC/generics/matrix.c \
    $PATH_SRC/generics/matrix_reader.c \
    $PATH_SRC/algorithms/wam.c \
    $PATH_SRC/algorithms/ddg.c \
    $PATH_SRC/algorithms/lnorm.c \
    $PATH_SRC/algorithms/jacobi.c \
    $PATH_SRC/algorithms/eigengap.c \
    $PATH_SRC/spkmeans.c \
    -lm \
    -o $PATH_OUT/spkmeans
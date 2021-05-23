#!/bin/sh

g++ -O2 LU_omp.cpp -o lu_omp -fopenmp

./lu_omp

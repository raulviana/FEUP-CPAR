#!/bin/sh

g++ -O2 LU_omp_tasks.cpp -o lu_omp_tasks -fopenmp

./lu_omp_tasks

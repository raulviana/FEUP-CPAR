#!/bin/sh

sudo sh -c "echo -1 > /proc/sys/kernel/perf_event_paranoid" 

g++ -O2 LU_omp.cpp -o lu_omp -fopenmp

./lu_omp

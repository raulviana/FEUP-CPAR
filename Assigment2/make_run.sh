#!/bin/sh

sudo sh -c "echo -1 > /proc/sys/kernel/perf_event_paranoid" 

g++ -O2 matrix_mul.cpp -o matrix_mul -lpapi

./matrix_mul
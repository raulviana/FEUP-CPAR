#!/bin/sh

sudo sh -c "echo -1 > /proc/sys/kernel/perf_event_paranoid" 

g++ -O2 matrix.cpp -o matrix -lpapi

./matrix

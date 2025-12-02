#!/bin/bash
# Script para generar outputs de tests

# Test ReLU 2
cd tests/relu/test_2
g++ -std=c++20 -I../.. test_2.cpp ../../test-main.o -o test_2_gen 2>/dev/null
echo "" | ./test_2_gen 2>&1 | grep -A 100 "Valores Obtenidos" | grep -A 50 "Caso #1" | tail -n +2 | head -18

cd ../../..

#!/bin/bash
echo "====================================="
echo "Ejecutando Tests Unitarios"
echo "====================================="
echo ""

total=0
passed=0

for category in relu dense; do
    echo "Categoría: $category"
    for i in 1 2 3 4; do
        cd tests/$category/test_$i
        result=$(./run_test 2>&1 | grep -E "All tests passed")
        if [ -n "$result" ]; then
            echo "  ✓ Test $i: PASÓ"
            ((passed++))
        else
            echo "  ✗ Test $i: FALLÓ"
        fi
        ((total++))
        cd ../../..
    done
    echo ""
done

echo "Resumen: $passed/$total tests pasaron"

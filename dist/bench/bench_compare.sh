#!/usr/bin/env bash

# Usage: ./bench_compare.sh
# This script benchmarks both PHP (ort) and Python (numpy)
# Results are written to :
#   bench_php_{function}.csv
#   bench_py_{function}.csv

# Allow environment overrides, with sensible defaults
PHP="${PHP:-php}"
PYTHON="${PYTHON:-python3}"
BENCH_PHP="${BENCH_PHP:-bench.php}"
BENCH_PY="${BENCH_PY:-bench.py}"
BENCH_REPEAT="${BENCH_REPEAT:-10}"

# Allow override for core and size lists
BENCH_CORES_LIST="${BENCH_CORES_LIST:-4 8 12 16 24}"
BENCH_SIZE_LIST="${BENCH_SIZE_LIST:-1024 2048 4096 8192}"
BENCH_FUNCTION_LIST="${BENCH_FUNCTION_LIST:-add sub mul div}"
BENCH_CSV="${BENCH_CSV:-csv}"

for BENCH_FUNCTION in $BENCH_FUNCTION_LIST; do
  BENCH_FUNCTION=$(echo "$BENCH_FUNCTION" | tr -d '\n')
  BENCH_OUT_PHP="$BENCH_CSV/php.$BENCH_FUNCTION.csv"
  BENCH_OUT_PY="$BENCH_CSV/py.$BENCH_FUNCTION.csv"

  # CSV headers (output both mean and trimmed mean columns)
  printf "cores,size,function,mean_time,trimmed_mean_time,mops,trimmed_mops\n" > $BENCH_OUT_PHP
  printf "cores,size,function,mean_time,trimmed_mean_time,mops,trimmed_mops\n" > $BENCH_OUT_PY

  # Main loop, using env overrides for core/size lists
  for BENCH_CORES in $BENCH_CORES_LIST; do
    for BENCH_SIZE in $BENCH_SIZE_LIST; do
      echo "Benchmarking $PHP function: $BENCH_FUNCTION with $BENCH_CORES cores and size $BENCH_SIZE ($BENCH_REPEAT times)"
      # PHP/ort: process isolation, JSON output, extract all relevant fields
      PHP_JSON_OUT=$(ORT_SCALE_CORES=$BENCH_CORES $PHP -dextension=ort -dmemory_limit=-1 "$BENCH_PHP" "$BENCH_FUNCTION" "$BENCH_SIZE" "$BENCH_REPEAT" --json 2>/dev/null)
      if echo "$PHP_JSON_OUT" | grep -q '"trimmed_mean_time"'; then
        PHP_CSV_LINE=$(python3 -c "import sys, json; d=json.load(sys.stdin); print(f'{d.get('mean_time','')},{d.get('trimmed_mean_time','')},{d.get('mops','')},{d.get('trimmed_mops','')}')" <<< "$PHP_JSON_OUT")
        printf "%s,%s,%s,%s\n" "$BENCH_CORES" "$BENCH_SIZE" "$BENCH_FUNCTION" "$PHP_CSV_LINE" >> $BENCH_OUT_PHP
      else
        echo "ORT_SCALE_CORES=$BENCH_CORES $PHP -dmemory_limit=-1 "$BENCH_PHP" "$BENCH_FUNCTION" "$BENCH_SIZE" "$BENCH_REPEAT" --json FAILED: \"$PHP_JSON_OUT\"" >&2
        printf "%s,%s,%s,,,,\n" "$BENCH_CORES" "$BENCH_SIZE" "$BENCH_FUNCTION" >> $BENCH_OUT_PHP
      fi

      echo "Benchmarking $PYTHON function: $BENCH_FUNCTION with $BENCH_CORES cores and size $BENCH_SIZE ($BENCH_REPEAT times)"
      # Python/numpy: process isolation, JSON output, extract all relevant fields
      PY_JSON_OUT=$(OMP_NUM_THREADS=$BENCH_CORES $PYTHON "$BENCH_PY" "$BENCH_FUNCTION" "$BENCH_SIZE" "$BENCH_REPEAT" --json 2>/dev/null)
      if echo "$PY_JSON_OUT" | grep -q '"trimmed_mean_time"'; then
        PY_CSV_LINE=$(python3 -c "import sys, json; d=json.load(sys.stdin); print(f'{d.get('mean_time','')},{d.get('trimmed_mean_time','')},{d.get('mops','')},{d.get('trimmed_mops','')}')" <<< "$PY_JSON_OUT")
        printf "%s,%s,%s,%s\n" "$BENCH_CORES" "$BENCH_SIZE" "$BENCH_FUNCTION" "$PY_CSV_LINE" >> $BENCH_OUT_PY
      else
        echo "OMP_NUM_THREADS=$BENCH_CORES $PYTHON "$BENCH_PY" "$BENCH_FUNCTION" "$BENCH_SIZE" "$BENCH_REPEAT" --json FAILED: \"$PY_JSON_OUT\"" >&2
        printf "%s,%s,%s,,,,\n" "$BENCH_CORES" "$BENCH_SIZE" "$BENCH_FUNCTION" >> $BENCH_OUT_PY
      fi
    done
  done

  echo "Benchmark $BENCH_FUNCTION complete:"
  echo "  PHP=$PHP"
  echo "  PYTHON=$PYTHON"
  echo "  BENCH_PHP=$BENCH_PHP"
  echo "  BENCH_PY=$BENCH_PY"
  echo "  BENCH_FUNCTION=$BENCH_FUNCTION"
  echo "  BENCH_REPEAT=$BENCH_REPEAT"
  echo "  BENCH_CORES_LIST=$BENCH_CORES_LIST"
  echo "  BENCH_SIZE_LIST=$BENCH_SIZE_LIST"
  echo "Results in $BENCH_OUT_PHP and $BENCH_OUT_PY."
done

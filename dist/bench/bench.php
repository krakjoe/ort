<?php
// Usage: php bench.php <func> <size> [repeat=10]

$func = getenv('BENCH_FUNCTION') ?: ($argv[1] ?? null);
$size = getenv('BENCH_SIZE') ?: ($argv[2] ?? null);
$repeat = getenv('BENCH_REPEAT') ?: ($argv[3] ?? 10);
$as_json = (getenv('BENCH_JSON') && in_array(strtolower(getenv('BENCH_JSON')), ['1','true','yes'])) || in_array('--json', $argv);
if (!$func || !$size) {
    fwrite(STDERR, "Usage: php bench.php <func> <size> [repeat] [--json]\n");
    exit(1);
}
$size = (int)$size;
$repeat = (int)$repeat;

if (!extension_loaded('ort')) {
    fwrite(STDERR, "ort extension not loaded\n");
    exit(1);
}

$dataA = [];
$dataB = [];
for ($i = 0; $i < $size; $i++) {
    $rowA = [];
    $rowB = [];
    for ($j = 0; $j < $size; $j++) {
        $rowA[] = mt_rand() / mt_getrandmax();
        $rowB[] = mt_rand() / mt_getrandmax();
    }
    $dataA[] = $rowA;
    $dataB[] = $rowB;
}

$a = ORT\Tensor\Transient::from($dataA, ORT\Tensor::FLOAT32);
$b = ORT\Tensor\Transient::from($dataB, ORT\Tensor::FLOAT32);

$php_funcs = [
    'add' => 'ORT\\Math\\add',
    'mul' => 'ORT\\Math\\multiply',
    'sub' => 'ORT\\Math\\subtract',
    'div' => 'ORT\\Math\\divide',
];

if (!isset($php_funcs[$func]) || !function_exists($php_funcs[$func])) {
    fwrite(STDERR, "Unknown function: $func\n");
    exit(1);
}

if (!$as_json) {
    printf("%s([%d x %d], [%d x %d]) %d threads, %s backend\n",
        $func,
        $size, $size,
        $size, $size,
        \ORT\Math\scale\cores(),
        ($backends = \ORT\Math\backend()) ?
            \implode(", ", $backends) : 'scalar');
}

# Warmup
for ($r = 0; $r < $repeat; $r++) {
    $result = $php_funcs[$func]($a, $b);
}

$times = [];
for ($r = 0; $r < $repeat; $r++) {
    $start = microtime(true);
    $result = $php_funcs[$func]($a, $b);
    $elapsed = microtime(true) - $start;
    $times[] = $elapsed;
    if (!$as_json) {
        printf("run %d/%d: time=%.6fs\n", $r+1, $repeat, $elapsed);
    }
}

function trimmed_mean(array $arr, int $trim = 1): float {
    $n = count($arr);
    if ($n <= 2 * $trim) return array_sum($arr) / $n;
    sort($arr, SORT_NUMERIC);
    $arr = array_slice($arr, $trim, $n - 2 * $trim);
    return array_sum($arr) / count($arr);
}
$trim = min(1, intdiv($repeat, 4));
$mean_time = array_sum($times) / count($times);
$trimmed = trimmed_mean($times, $trim);
$sum = ORT\Math\reduce\tensor\sum($result)->getData()[0];
$ops = $size * $size;
$mops = $ops / $mean_time / 1e6;
$trimmed_mops = $ops / $trimmed / 1e6;

if ($as_json) {
    $out = [
        'function' => $func,
        'size' => $size,
        'repeat' => $repeat,
        'times' => $times,
        'mean_time' => $mean_time,
        'trimmed_mean_time' => $trimmed,
        'mops' => $mops,
        'trimmed_mops' => $trimmed_mops,
        'sum' => $sum,
        'threads' => function_exists('ORT\\Math\\cores') ? \ORT\Math\cores() : -1,
        'backend' => function_exists('ORT\\Math\\backend') ? (\ORT\Math\backend() ?: 'scalar') : 'unknown',
    ];
    echo json_encode($out, JSON_UNESCAPED_SLASHES|JSON_UNESCAPED_UNICODE|JSON_PRETTY_PRINT) . "\n";
} else {
    printf("sum=%.6f\n", $sum);
    printf("mean_time=%.6fs\n", $mean_time);
    printf("trimmed_mean_time=%.6fs\n", $trimmed);
    printf("mops=%.2f\n", $mops);
    printf("trimmed_mops=%.2f\n", $trimmed_mops);
}
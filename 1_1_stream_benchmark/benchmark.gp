# ── Stream Benchmark Plot ────────────────────────────────────────────────────
set terminal pngcairo size 1200,700 enhanced font "Arial,13"
set output "results/benchmark.png"

set title "Memory Bandwidth vs Array Size" font "Arial,16"
set xlabel "Array Size (elements)"
set ylabel "Bandwidth (GB/s)"

set logscale x 2
set format x "2^{%L}"
set grid xtics ytics lt 0 lw 1 lc rgb "#cccccc"

set key top left box opaque

set style line 1 lc rgb "#2196F3" lw 2 pt 7 ps 1.2   # base  - blue
set style line 2 lc rgb "#4CAF50" lw 2 pt 9 ps 1.2   # omp   - green
set style line 3 lc rgb "#F44336" lw 2 pt 5 ps 1.2   # cuda  - red

plot "results/benchmark.dat" using 1:2 with linespoints ls 1 title "stream-base",      "results/benchmark.dat" using 1:3 with linespoints ls 2 title "stream-omp-host",      "results/benchmark.dat" using 1:4 with linespoints ls 3 title "stream-cuda"

set title "DGEMM - Intel® Xeon® Processor E3-1245 v2 (8M Cache, 3.40 GHz)"
set terminal png size 1200,900
set output 'time.png'
set autoscale
set grid linetype 0
#set xrange [0:1]
set yrange [0:0.2]
set xlabel "Matrix size m=n=k"
set ylabel "Time in s"
set key width -25 Left reverse left top autotitle columnheader noenhanced
plot for [idx=2:100] 'time.dat' using 1:idx with linespoints
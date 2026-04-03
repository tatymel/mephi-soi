# mephi-soi

source /opt/intel/oneapi/setvars.sh


icpx -fsycl -O3 -march=native -mavx2 main.cpp -o median_filter

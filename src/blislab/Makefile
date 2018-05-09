CC = icc
CFLAGS = -O3 -fopenmp -mavx -fPIC -I.
LDLIBS = -lpthread -lm -qopenmp -mkl=parallel -Werror -Wall -pedantic

test:
	$(CC) $(CFLAGS) dgemm_baseline.cpp $(LDLIBS) -o dgemm_baseline.x
	$(CC) $(CFLAGS) dgemm_optimized.cpp $(LDLIBS) -o dgemm_optimized.x

clean:
	@rm -f dgemm_baseline.x
	@rm -f dgemm_optimized.x

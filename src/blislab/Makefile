CC = icc
CFLAGS = -O3 -fopenmp -mavx -fPIC -I.
LDLIBS = -lpthread -lm -qopenmp -mkl=parallel -Werror -Wall -pedantic

test:
	$(CC) $(CFLAGS) dgemm_baseline.cpp $(LDLIBS) -o dgemm_baseline.x

clean:
	@rm -f dgemm_baseline.x

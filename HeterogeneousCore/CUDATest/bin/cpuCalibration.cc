#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "tbb/tick_count.h"

namespace {
  unsigned long findPrimes(const unsigned n_iterations) {
    // Flag to trigger the allocation
    bool is_prime;
    // Let's prepare the material for the allocations
    unsigned int primes_size = 1;
    std::vector<unsigned long> primes(primes_size);
    primes[0] = 2;
    unsigned long i = 2;

    // Loop on numbers
    for (unsigned long int iiter = 0; iiter < n_iterations; iiter++) {
      // Once at max, it returns to 0
      i += 1;
      // Check if it can be divided by the smaller ones
      is_prime = true;
      for (unsigned long j = 2; j < i && is_prime; ++j) {
        if (i % j == 0)
          is_prime = false;
      }  // end loop on numbers < than tested one

      if (is_prime) {
        // copy the array of primes (INEFFICIENT ON PURPOSE!)
        unsigned int new_primes_size = 1 + primes_size;
        std::vector<unsigned long> new_primes(new_primes_size);
        for (unsigned int prime_index = 0; prime_index < primes_size; prime_index++) {
          new_primes[prime_index] = primes[prime_index];
        }
        // attach the last prime
        new_primes[primes_size] = i;
        // Update primes array
        std::swap(primes, new_primes);
        primes_size = new_primes_size;
      }  // end is prime
    }    // end of while loop

    if (primes.empty()) {
      return 0;
    }
    return primes.back();
  }
}  // namespace

int main(int argc, char **argv) {
  std::vector<size_t> iters{0,     10,    30,    50,    70,    100,   200,    300,    400,   500,   600,
                            700,   800,   1000,  1300,  1600,  2000,  2300,   2600,   3000,  3300,  3500,
                            3900,  4200,  5000,  6000,  8000,  10000, 12000,  15000,  17000, 20000, 25000,
                            30000, 35000, 40000, 50000, 60000, 80000, 100000, 150000, 200000};

  std::vector<double> times(iters.size(), 0);

  // warm it up by doing 20k iterations
  findPrimes(20000);

  std::ofstream out("cpuCalibration.json");
  std::string prefix = "  ";
  out << "{\n" << prefix << "\"niters\": [\n";
  for (auto i : iters) {
    out << prefix << prefix << i;
    if (i != iters.back()) {
      out << ",";
    }
    out << "\n";
  }
  out << prefix << "],\n" << prefix << "\"timesInMicroSeconds\": [\n" << prefix << prefix << "0.0,\n";

  for (unsigned int i = 1; i < iters.size(); ++i) {
    unsigned int niters = iters[i];

    auto start_cali = tbb::tick_count::now();
    findPrimes(niters);
    auto stop_cali = tbb::tick_count::now();
    auto deltat = (stop_cali - start_cali).seconds();
    times[i] = deltat * 1000000.;  // in microseconds
    std::cout << " Calibration: # iters = " << niters << " => " << times[i] << " us" << std::endl;
    out << prefix << prefix << times[i];
    if (i + 1 != iters.size()) {
      out << ",";
    }
    out << "\n";
  }

  out << prefix << "]\n"
      << "}\n";
}

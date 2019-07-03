#ifndef HeterogeneousCore_CUDATest_TimeCruncher_h
#define HeterogeneousCore_CUDATest_TimeCruncher_h

#include "tbb/tick_count.h"

#include <chrono>

// The contents of this class are from
// https://gitlab.cern.ch/gaudi/Gaudi/blob/master/GaudiSvc/src/CPUCrunchSvc/CPUCrunchSvc.cpp
namespace cudatest {
  inline
  unsigned long findPrimes(const unsigned n_iterations ) {
    // Flag to trigger the allocation
    bool is_prime;
    // Let's prepare the material for the allocations
    unsigned int   primes_size = 1;
    std::vector<unsigned long> primes(primes_size);
    primes[0]                  = 2;
    unsigned long i = 2;

    // Loop on numbers
    for ( unsigned long int iiter = 0; iiter < n_iterations; iiter++ ) {
      // Once at max, it returns to 0
      i += 1;
      // Check if it can be divided by the smaller ones
      is_prime = true;
      for ( unsigned long j = 2; j < i && is_prime; ++j ) {
        if ( i % j == 0 ) is_prime = false;
      } // end loop on numbers < than tested one

      if ( is_prime ) {
        // copy the array of primes (INEFFICIENT ON PURPOSE!)
        unsigned int   new_primes_size = 1 + primes_size;
        std::vector<unsigned long> new_primes(new_primes_size);
        for ( unsigned int prime_index = 0; prime_index < primes_size; prime_index++ ) {
          new_primes[prime_index] = primes[prime_index];
        }
        // attach the last prime
        new_primes[primes_size] = i;
        // Update primes array
        std::swap(primes, new_primes);
        primes_size = new_primes_size;
      } // end is prime
    } // end of while loop

    if(primes.empty()) {
      return 0;
    }
    return primes.back();
  }

  /*
    Calibrate the crunching finding the right relation between max number to be searched and time spent.
    The relation is a sqrt for times greater than 10^-4 seconds.
  */
  class TimeCruncher {
  public:
    TimeCruncher() {
      times_vect_.resize(niters_vect_.size());
      times_vect_[0] = 0;

      // warm it up by doing 20k iterations
      findPrimes( 20000 );

      for ( unsigned int i = 1; i < niters_vect_.size(); ++i ) {
        unsigned int niters = niters_vect_[i];
        unsigned int trials = 30;
        do {
          auto start_cali = tbb::tick_count::now();
          findPrimes( niters );
          auto stop_cali       = tbb::tick_count::now();
          auto deltat          = ( stop_cali - start_cali ).seconds();
          times_vect_[i] = deltat * 1000000.; // in microseconds
          LogTrace("foo") << " Calibration: # iters = " << niters << " => " << times_vect_[i] << " us";
          trials--;
        } while ( trials > 0 && times_vect_[i] < times_vect_[i-1] ); // make sure that they are monotonic
      }
    }

    TimeCruncher(const std::vector<unsigned int>& iters, const std::vector<double>& times) {
      if(iters.size() != times.size()) {
        throw cms::Exception("Configuration") << "CPU Calibration: got " << iters.size() << " iterations and " << times.size() << " times";
      }
      if(iters.empty()) {
        throw cms::Exception("Configuration") << "CPU Calibration: iterations is empty";
      }
      if(times.empty()) {
        throw cms::Exception("Configuration") << "CPU Calibration: times is empty";
      }

      niters_vect_ = iters;
      times_vect_ = times;
    }

    void crunch_for(const std::chrono::nanoseconds& crunchtime) const {
      const unsigned int niters = getNCaliIters( crunchtime );
      auto start_cali = tbb::tick_count::now();
      findPrimes( niters );
      auto stop_cali = tbb::tick_count::now();

      std::chrono::nanoseconds actual( int( 1e9 * ( stop_cali - start_cali ).seconds() ) );

      LogTrace("foo") << "crunch for " << (crunchtime.count()*1e-3) << " us == " << niters << " iter. actual time: " << (actual.count()*1e-3)
                      << " us. ratio: " << float( actual.count() ) / crunchtime.count();


    }

  private:
    unsigned int getNCaliIters( const std::chrono::nanoseconds& runtime ) const {
      unsigned int smaller_i   = 0;
      double       time        = 0.;
      bool         found       = false;
      double       corrRuntime = runtime.count()/1000.; // * m_corrFact;
      // We know that the first entry is 0, so we start to iterate from 1
      for ( unsigned int i = 1; i < times_vect_.size(); i++ ) {
        time = times_vect_[i];
        if ( time > corrRuntime ) {
          smaller_i = i - 1;
          found     = true;
          break;
        }
      }
      // Case 1: we are outside the interpolation range, we take the last 2 points
      if ( not found ) smaller_i = times_vect_.size() - 2;
      // Case 2: we maeke a linear interpolation
      // y=mx+q
      const auto   x0 = times_vect_[smaller_i];
      const auto   x1 = times_vect_[smaller_i + 1];
      const auto   y0 = niters_vect_[smaller_i];
      const auto   y1 = niters_vect_[smaller_i + 1];
      const double m  = (double)( y1 - y0 ) / (double)( x1 - x0 );
      const double q  = y0 - m * x0;
      const unsigned int nCaliIters = m * corrRuntime + q;
      LogDebug("foo") << "x0: " << x0 << " x1: " << x1 << " y0: " << y0 << " y1: " << y1 << "  m: " << m << " q: " << q
                  << "  itr: " << nCaliIters;
      return nCaliIters;
    }

    std::vector<unsigned int> niters_vect_ = {
      0, 10, 30, 50, 70, 100, 200, 300, 400, 500,   600,   700,   800,   1000,  1300,  1600,  2000,  2300,
      2600,  3000,  3300,  3500,  3900,  4200,  5000,  6000,  8000,  10000,
      12000, 15000, 17000, 20000, 25000, 30000, 35000, 40000, 50000, 60000
      // long calib
      //80000, 100000, 150000, 200000
    };
    std::vector<double> times_vect_; // in us
  };

  inline
  const TimeCruncher& getTimeCruncher() {
    const static TimeCruncher cruncher;
    return cruncher;
  }
}

#endif


//#include <sys/resource.h>
//#include <sys/time.h>
#include <iostream>

class Timer
{
  public:
    //! Resets and starts timer
    inline void start();

    //! Stops timer and gives elapsed time in seconds
    inline void stop();
  private:
		//timeval t1, t2;
		//rusage r1, r2;
};

inline void Timer::start()
{
	//gettimeofday( &t1, (struct timezone*)0 );
	//getrusage( RUSAGE_SELF, &r1 );

	//t2 = t1;
	//r2 = r1;
}

inline void Timer::stop()
{
	//gettimeofday( &t2, (struct timezone*)0 );
	//getrusage( RUSAGE_SELF, &r2 );

	//double user = double(r2.ru_utime.tv_sec - r1.ru_utime.tv_sec) + double(r2.ru_utime.tv_usec - r1.ru_utime.tv_usec) / 1000000.0; 
	//double sys = double(r2.ru_stime.tv_sec - r1.ru_stime.tv_sec) + double(r2.ru_stime.tv_usec - r1.ru_stime.tv_usec) / 1000000.0; 
	//double real = double(t2.tv_sec - t1.tv_sec) + double(t2.tv_usec - t1.tv_usec) / 1000000.0;

	//std::cout << "u: " << user << " s: " << sys << " r: " << 100.0 * ( user + sys ) / real << std::endl;
}

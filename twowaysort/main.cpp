#include <iostream>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <fstream>
#include <vector>
#include <algorithm>
#define CHUNKSIZE 4217
#define NUMCHUNKS 3951
#define BLOCKSIZE 524288

//#define DBG 1
typedef boost::minstd_rand base_generator_type;

#include "tape.h"
#include "twoWayMerge.h"

void createLargeFile()
{
	base_generator_type generator(42u);
	
	boost::uniform_real<> uni_dist(0,100);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);

	std::cout.setf(std::ios::fixed);
	
	std::fstream filestr;

	filestr.open ("./tapea1.b", std::fstream::out | std::fstream::binary);
	filestr.close();
	
	float *chunk = new float[CHUNKSIZE];
	for(int j = 0; j < NUMCHUNKS; j++) {
		for(int i = 0; i < CHUNKSIZE; i++) {
			chunk[i] = uni();
			
		}
		filestr.open ("./tapea1.b", std::fstream::out | std::fstream::binary | std::fstream::app);
		filestr.write((char*)chunk, sizeof(float) * CHUNKSIZE);
		filestr.close();
	}
	delete[] chunk;
	
	std::cout << "written "<<NUMCHUNKS<<" chunks\n";
}

int main (int argc, char * const argv[]) {
    std::cout << "External Sort!\nstart writing a large file... ";
	
	createLargeFile();
	
	TwoWayMerge merg;
	merg.doIt("./tapea1.b", 4);
	merg.validate();
    return 0;
}

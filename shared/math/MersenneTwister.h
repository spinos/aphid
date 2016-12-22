/*
 *  MersenneTwister.h
 *  aphid
 *
 *  Created by jian zhang on 8/16/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef MERSENNETWISTER_H
#define MERSENNETWISTER_H
namespace aphid {
typedef   signed long int32;     
typedef unsigned long uint32;
class MersenneTwister {
	#define MERS_N   624
    #define MERS_M   397
    #define MERS_R   31
    #define MERS_U   11
    #define MERS_S   7
    #define MERS_T   15
    #define MERS_L   18
    #define MERS_A   0x9908B0DF
    #define MERS_B   0x9D2C5680
    #define MERS_C   0xEFC60000
    
public:
	MersenneTwister(uint32 seed);
	virtual ~MersenneTwister();
	
	double random();
	uint32 bRandom();
	int iRandom(int min, int max);
	
protected:
	void randomInit(uint32 seed);        // re-seed
	
private:
	uint32 mt[MERS_N];                   // state vector
	int mti;
		
	enum TArch {
		achLITTLEENDIAN = 0, 
		achBIGENDIAN = 1, 
		achNONIEEE = 2
	};
	
	TArch Architecture; 
};
}
#endif
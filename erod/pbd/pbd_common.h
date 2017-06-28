#ifndef PBD_COMMON_H
#define PBD_COMMON_H
#include <AllMath.h>
namespace pbd {

struct Spring {
		unsigned p1, p2;
		float rest_length;
		float Ks, Kd;
		int type;
};

struct DistanceConstraint {	
	unsigned p1, p2;	
	float rest_length, k_prime; 
};

struct BendingConstraint {	
	unsigned p1, p2, p3;	
	float rest_length,  w,  k, k_prime;
};

}
#endif        //  #ifndef PBD_COMMON_H


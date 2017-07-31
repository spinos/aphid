/*
 *  UniformGrid8Sphere.h
 *  [-1, 1] 8 x 8 x 8 grid cell centers within 1
 *  512 cells 280 samples
 *
 *  Created by jian zhang on 8/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_SMP_UNIFORMGRID8_SPHERE_H
#define APH_SMP_UNIFORMGRID8_SPHERE_H

namespace aphid {

class Vector3F;

namespace smp {

class UniformGrid8Sphere {

	static const int sGridToSampleTable[];
	
public:
	UniformGrid8Sphere();
	virtual ~UniformGrid8Sphere();
	
/// pv within [-1, 1]
	static int GetSampleInd(const Vector3F& pv);
/// closest cell center
	static Vector3F GetSamplePnt(const Vector3F& pv);
	
	static const int sNumSamples;
/// sample points
	static const float sSamplePnts[][3];
	static const float sCellSize;
	
private:
	static void GetSampleCoord(int* dst, const Vector3F& pv);
	
};

}
}

#endif

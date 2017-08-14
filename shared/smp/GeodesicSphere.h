/*
 *  GeodesicSphere.h
 *
 *  directional samples at each face center of a geodesic dome
 *  36 samples when level = 3
 *
 *  Created by jian zhang on 8/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_SMP_GEODESIC_SPHERE_H
#define APH_SMP_GEODESIC_SPHERE_H

namespace aphid {

class Vector3F;

namespace smp {

class GeodesicSphere {

	Vector3F* m_samplePnts;
	int m_numSamples;
	
public:
	GeodesicSphere();
	virtual ~GeodesicSphere();
	
	void generateSamples(const float& angleLimit = 1.57f, int level=3);
	const int& numSamples() const;
	const Vector3F& getSample(int i) const;
	Vector3F getSample(int i, float noi) const;
	
private:
	
};

}
}

#endif

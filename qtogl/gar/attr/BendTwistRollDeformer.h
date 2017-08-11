/*
 *  BendTwistRollDeformer.h
 *  
 *  bend effect > twist effect > roll
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef BEND_TWIST_ROLL_DEFORMER_H
#define BEND_TWIST_ROLL_DEFORMER_H

#include <boost/scoped_array.hpp>

namespace aphid {

class Vector3F;
class Matrix33F;
class ATriangleMesh;
}


class BendTwistRollDeformer {

	const aphid::ATriangleMesh * m_mesh;
	boost::scoped_array<aphid::Vector3F > m_points;
	boost::scoped_array<aphid::Vector3F > m_normals;

	float m_warpAngle[2];
	
public:
	BendTwistRollDeformer(const aphid::ATriangleMesh * mesh);
	virtual ~BendTwistRollDeformer();
	
	void setWarpAngles(const float * v);
	
	void deform(const aphid::Matrix33F & mat);
	void calculateNormal();

	const aphid::Vector3F * deformedPoints() const;
	const aphid::Vector3F * deformedNormals() const;
	
protected:
	
private:
};
#endif

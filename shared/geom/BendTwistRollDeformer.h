/*
 *  BendTwistRollDeformer.h
 *  
 *  deform a billboard
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

class BendTwistRollDeformer {

	boost::scoped_array<Vector3F > m_points;
	boost::scoped_array<Vector3F > m_normals;
/// num point to deform
	int m_np;
/// bend-x, twist-y, roll-z rotation
	float m_angles[3];
	
public:
    BendTwistRollDeformer();
	virtual ~BendTwistRollDeformer();
	
	void setBend(const float& x);
	void setTwist(const float& x);
	void setRoll(const float& x);
	
	void deform(const ATriangleMesh * mesh);
	
	const Vector3F * deformedPoints() const;
	const Vector3F * deformedNormals() const;
	
protected:
	
private:
	float getRowMean(int rowBegin, int nv, int& nvRow, float& rowBase ) const;
	void setOriginalMesh(const ATriangleMesh * mesh);
	void calculateNormal(const ATriangleMesh * mesh);

};

}
#endif

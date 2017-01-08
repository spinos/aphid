/*
 *  MAvianArm.h
 *  cinchona
 *
 *  Created by jian zhang on 1/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef M_AVIAN_ARM_H
#define M_AVIAN_ARM_H
#include <maya/MObject.h>
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include "DrawAvianArm.h"
#include <vector>

class MAvianArm : public DrawAvianArm {

public:
	MAvianArm();
	virtual ~MAvianArm();
	
protected:
	void setSkeletonMatrices(const MObject & node,
					const MObject & humerusAttr, 
					const MObject & ulnaAttr, 
					const MObject & radiusAttr, 
					const MObject & carpusAttr, 
					const MObject & secondDigitAttr);
	
	void setLigamentParams(const MObject & node,
					const MObject & lig0xAttr,
					const MObject & lig0yAttr,
					const MObject & lig0zAttr,
					const MObject & lig1xAttr,
					const MObject & lig1yAttr,
					const MObject & lig1zAttr );
					
	void setElbowParams(const MObject & node,
					const MObject & elbowxAttr,
					const MObject & elbowyAttr,
					const MObject & elbowzAttr);
					
	void setWristParams(const MObject & node,
					const MObject & x0Attr,
					const MObject & y0Attr,
					const MObject & z0Attr,
					const MObject & x1Attr,
					const MObject & y1Attr,
					const MObject & z1Attr);
					
	void set2ndDigitParams(const MObject & node,
					const MObject & x0Attr,
					const MObject & y0Attr,
					const MObject & z0Attr,
					const MObject & lAttr);
					
	void setFirstLeadingLigament();
	
	void setFlyingFeatherGeomParam(const MObject & node,
					const MObject & n0Attr,
					const MObject & n1Attr,
					const MObject & n2Attr,
					const MObject & c0Attr,
					const MObject & c1Attr,
					const MObject & c2Attr,
					const MObject & c3Attr,
					const MObject & t0Attr,
					const MObject & t1Attr,
					const MObject & t2Attr,
					const MObject & t3Attr);
					
	void setFeatherOrientationParam(const MObject & node,
					const MObject & m0Attr,
					const MObject & m1Attr,
					const MObject & m2Attr,
					const MObject & u0rzAttr);
					
	void setFeatherDeformationParam(const MObject & node, 
					const MObject & brt0Attr, 
					const MObject & brt1Attr, 
					const MObject & brt2Attr, 
					const MObject & brt3Attr);
/// i 1:2 upper
/// i 3:4 lower					
	void setCovertFeatherGeomParam(int i,
					const MObject & node,
					const MObject & n0Attr,
					const MObject & n1Attr,
					const MObject & n2Attr,
					const MObject & n3Attr,
					const MObject & c0Attr,
					const MObject & c1Attr,
					const MObject & c2Attr,
					const MObject & c3Attr,
					const MObject & c4Attr,
					const MObject & t0Attr,
					const MObject & t1Attr,
					const MObject & t2Attr,
					const MObject & t3Attr,
					const MObject & t4Attr);
	
private:

};
#endif
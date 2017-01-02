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
					const MObject & x1Attr,
					const MObject & y1Attr,
					const MObject & z1Attr,
					const MObject & lAttr);
	void setFirstLeadingLigament();
	
private:

};
#endif
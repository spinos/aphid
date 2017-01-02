/*
 *  MAvianArm.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "MAvianArm.h"
#include <AHelper.h>

using namespace aphid;

MAvianArm::MAvianArm()
{}

MAvianArm::~MAvianArm()
{}

void MAvianArm::setSkeletonMatrices(const MObject & node,
					const MObject & humerusAttr, 
					const MObject & ulnaAttr, 
					const MObject & radiusAttr, 
					const MObject & carpusAttr, 
					const MObject & secondDigitAttr)
{
	MMatrix humerusM = AHelper::getMatrixAttr(node, humerusAttr);
	MMatrix ulnaM = AHelper::getMatrixAttr(node, ulnaAttr);
	ulnaM *= humerusM;
	MMatrix radiusM = AHelper::getMatrixAttr(node, radiusAttr);
	radiusM *= ulnaM;
	MMatrix carpusM = AHelper::getMatrixAttr(node, carpusAttr);
	carpusM *= radiusM;
	MMatrix secondDigitM = AHelper::getMatrixAttr(node, secondDigitAttr);
	secondDigitM *= carpusM;
	
	AHelper::ConvertToMatrix44F(skeletonMatricesR()[0], humerusM);
	AHelper::ConvertToMatrix44F(skeletonMatricesR()[1], ulnaM);
	AHelper::ConvertToMatrix44F(skeletonMatricesR()[2], radiusM);
	AHelper::ConvertToMatrix44F(skeletonMatricesR()[3], carpusM);
	AHelper::ConvertToMatrix44F(skeletonMatricesR()[4], secondDigitM);

}

void MAvianArm::setLigamentParams(const MObject & node,
					const MObject & lig0xAttr,
					const MObject & lig0yAttr,
					const MObject & lig0zAttr,
					const MObject & lig1xAttr,
					const MObject & lig1yAttr,
					const MObject & lig1zAttr )
{
	Vector3F offset0;
	offset0.x = MPlug(node, lig0xAttr).asFloat();
	offset0.y = MPlug(node, lig0yAttr).asFloat();
	offset0.z = MPlug(node, lig0zAttr).asFloat();
	
	setLeadingLigamentOffset(0, offset0);
	
	Vector3F offset1;
	offset1.x = MPlug(node, lig1xAttr).asFloat();
	offset1.y = MPlug(node, lig1yAttr).asFloat();
	offset1.z = MPlug(node, lig1zAttr).asFloat();
	
	setTrailingLigamentOffset(0, offset1);

}

void MAvianArm::setElbowParams(const MObject & node,
					const MObject & elbowxAttr,
					const MObject & elbowyAttr,
					const MObject & elbowzAttr)
{
	Vector3F offset1;
	offset1.x = MPlug(node, elbowxAttr).asFloat();
	offset1.y = MPlug(node, elbowyAttr).asFloat();
	offset1.z = MPlug(node, elbowzAttr).asFloat();
	
	setTrailingLigamentOffset(1, offset1);
}

void MAvianArm::setWristParams(const MObject & node,
					const MObject & x0Attr,
					const MObject & y0Attr,
					const MObject & z0Attr,
					const MObject & x1Attr,
					const MObject & y1Attr,
					const MObject & z1Attr)
{
	Vector3F offset0;
	offset0.x = MPlug(node, x0Attr).asFloat();
	offset0.y = MPlug(node, y0Attr).asFloat();
	offset0.z = MPlug(node, z0Attr).asFloat();
	
	setLeadingLigamentOffset(1, offset0);
	
	Vector3F offset1;
	offset1.x = MPlug(node, x1Attr).asFloat();
	offset1.y = MPlug(node, y1Attr).asFloat();
	offset1.z = MPlug(node, z1Attr).asFloat();
	
	setTrailingLigamentOffset(2, offset1);
}

void MAvianArm::set2ndDigitParams(const MObject & node,
					const MObject & x0Attr,
					const MObject & y0Attr,
					const MObject & z0Attr,
					const MObject & x1Attr,
					const MObject & y1Attr,
					const MObject & z1Attr,
					const MObject & lAttr)
{
	Vector3F offset0;
	offset0.x = MPlug(node, x0Attr).asFloat();
	offset0.y = MPlug(node, y0Attr).asFloat();
	offset0.z = MPlug(node, z0Attr).asFloat();
	
	setLeadingLigamentOffset(2, offset0);
	
	float digitL = MPlug(node, lAttr).asFloat();
	set2ndDigitLength(digitL);
	
	setLeadingLigamentOffset(3, Vector3F(0.f, 0.f, digitL * .15f) );
	
	Vector3F offset1;
	offset1.x = MPlug(node, x1Attr).asFloat();
	offset1.y = MPlug(node, y1Attr).asFloat();
	offset1.z = MPlug(node, z1Attr).asFloat();
	
	setTrailingLigamentOffset(3, offset1);
	
	setLeadingLigamentOffset(4, Vector3F(0.f, 0.f, -digitL * .15f) );
	
	
}

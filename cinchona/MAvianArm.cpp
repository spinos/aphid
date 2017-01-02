/*
 *  MAvianArm.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "MAvianArm.h"
#include "Ligament.h"
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
	
	const float l0 = offset0.length();
	
	offset0 = invPrincipleMatrixR()->transformAsNormal(offset0);
	
	setLeadingLigamentOffset(0, offset0 * l0);
	
	Vector3F offset1;
	offset1.x = MPlug(node, lig1xAttr).asFloat();
	offset1.y = MPlug(node, lig1yAttr).asFloat();
	offset1.z = MPlug(node, lig1zAttr).asFloat();
	
	const float l1 = offset1.length();
	
	offset1 = invPrincipleMatrixR()->transformAsNormal(offset1);
	
	setTrailingLigamentOffset(0, offset1 * l1);

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
	const float l0 = offset0.length();
	offset0 = handMatrixR()->transformAsNormal(offset0);
	
	setLeadingLigamentOffset(1, offset0 * l0);
	
	Vector3F offset1;
	offset1.x = MPlug(node, x1Attr).asFloat();
	offset1.y = MPlug(node, y1Attr).asFloat();
	offset1.z = MPlug(node, z1Attr).asFloat();
	const float l1 = offset1.length();
	offset1 = handMatrixR()->transformAsNormal(offset1);
	
	setTrailingLigamentOffset(2, offset1 * l1);
	
	Vector3F tgt0(1.f, 0.f, 0.f);
	
	tgt0 = handMatrixR()->transformAsNormal(tgt0);
	tgt0.normalize();
	
	setLeadingLigamentTangent(1, tgt0);
	setTrailingLigamentTangent(2, tgt0);
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
	const float l0 = offset0.length();
	offset0 = fingerMatrixR()->transformAsNormal(offset0);
	offset0.normalize();
	
	setLeadingLigamentOffset(2, offset0 * l0);
	
	float digitL = MPlug(node, lAttr).asFloat();
	set2ndDigitLength(digitL);
	
	Vector3F tip0(0.f, 0.f, 1.f);
	tip0 = secondDigitMatirxR()->transformAsNormal(tip0);
	tip0.normalize();
	tip0 = invPrincipleMatrixR()->transformAsNormal(tip0);
	tip0.normalize();
	
	setLeadingLigamentOffset(3, tip0 * (digitL * .15f) );
	
	Vector3F offset1;
	offset1.x = MPlug(node, x1Attr).asFloat();
	offset1.y = MPlug(node, y1Attr).asFloat();
	offset1.z = MPlug(node, z1Attr).asFloat();
	const float l1 = offset1.length();
	offset1 = fingerMatrixR()->transformAsNormal(offset1);
	offset1.normalize();
	
	setTrailingLigamentOffset(3, offset1 * l1);
	
	Vector3F tip1(0.f, 0.f,-1.f);
	tip1 = secondDigitMatirxR()->transformAsNormal(tip1);
	tip1.normalize();
	tip1 = invPrincipleMatrixR()->transformAsNormal(tip1);
	tip1.normalize();
	
	setTrailingLigamentOffset(4, tip1 * (digitL * .15f) );
	
	Vector3F tgt0(1.f, 0.f, 0.f);
	
	tgt0 = fingerMatrixR()->transformAsNormal(tgt0);
	tgt0.normalize();
	
	setLeadingLigamentTangent(2, tgt0);
	setTrailingLigamentTangent(3, tgt0);

	Vector3F tgt1(1.f, 0.f, 0.f);
	
	tgt1 = secondDigitMatirxR()->transformAsNormal(tgt1);
	tgt1.normalize();
	tgt1 = invPrincipleMatrixR()->transformAsNormal(tgt1);
	tgt1.normalize();
	
	setLeadingLigamentTangent(3, tgt1);
	setTrailingLigamentTangent(4, tgt1);
	
}

void MAvianArm::setFirstLeadingLigament()
{
	Vector3F tgt0 = elbowPosition() - shoulderPosition();
	const float humerusL = tgt0.length();
	tgt0 = invPrincipleMatrixR()->transformAsNormal(tgt0);
	tgt0.normalize();
	tgt0 *= humerusL * 1.4f;
	
	setLeadingLigamentTangent(0, tgt0);
	
	Vector3F tgt1 = wristPosition() - elbowPosition();
	const float ulnaL = tgt1.length();
	tgt1 = invPrincipleMatrixR()->transformAsNormal(tgt1);
	tgt1.normalize();
	tgt1 *= ulnaL * 1.4f;
	
	leadingLigamentR()->setKnotTangent(1, tgt1, 0);
	
}

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
#include "FeatherGeomParam.h"
#include "FeatherDeformParam.h"
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
	
	setLeadingLigamentOffset(3, tip0 * (digitL * .05f) );
	
	Vector3F tip1(0.f, 0.f,-1.f);
	tip1 = secondDigitMatirxR()->transformAsNormal(tip1);
	tip1.normalize();
	tip1 = invPrincipleMatrixR()->transformAsNormal(tip1);
	tip1.normalize();
	
	setTrailingLigamentOffset(3, tip1 * (digitL * .05f) );
	
	Vector3F tgt0(1.f, 0.f, 0.f);
	
	tgt0 = fingerMatrixR()->transformAsNormal(tgt0);
	tgt0.normalize();
	
	setLeadingLigamentTangent(2, tgt0);
	
	Vector3F tgt1(1.f, 0.f, 0.f);
	
	tgt1 = secondDigitMatirxR()->transformAsNormal(tgt1);
	tgt1.normalize();
	tgt1 = invPrincipleMatrixR()->transformAsNormal(tgt1);
	tgt1.normalize();
	
	setLeadingLigamentTangent(3, tgt1);
	setTrailingLigamentTangent(3, tgt1 * 4.f);
	
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

void MAvianArm::setFlyingFeatherGeomParam(const MObject & node,
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
					const MObject & t3Attr)
{
	int nps[3];
	nps[0] = MPlug(node, n0Attr).asInt() + 1;
	nps[1] = MPlug(node, n1Attr).asInt() + 1;
	nps[2] = MPlug(node, n2Attr).asInt() + 1;
	float chs[4];
	chs[0] = MPlug(node, c0Attr).asFloat();
	chs[1] = MPlug(node, c1Attr).asFloat();
	chs[2] = MPlug(node, c2Attr).asFloat();
	chs[3] = MPlug(node, c3Attr).asFloat();
	float thickness[4];
	thickness[0] = MPlug(node, t0Attr).asFloat();
	thickness[1] = MPlug(node, t1Attr).asFloat();
	thickness[2] = MPlug(node, t2Attr).asFloat();
	thickness[3] = MPlug(node, t3Attr).asFloat();
	FeatherGeomParam * param = featherGeomParameter();
	param->setFlying(nps, chs, thickness);
}

void MAvianArm::setCovertFeatherGeomParam(int i,
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
					const MObject & t4Attr)
{
	int nps[4];
	nps[0] = MPlug(node, n0Attr).asInt() + 1;
	nps[1] = MPlug(node, n1Attr).asInt() + 1;
	nps[2] = MPlug(node, n2Attr).asInt() + 1;
	nps[3] = MPlug(node, n3Attr).asInt() + 1;
	float chs[5];
	chs[0] = MPlug(node, c0Attr).asFloat();
	chs[1] = MPlug(node, c1Attr).asFloat();
	chs[2] = MPlug(node, c2Attr).asFloat();
	chs[3] = MPlug(node, c3Attr).asFloat();
	chs[4] = MPlug(node, c4Attr).asFloat();
	float thickness[5];
	thickness[0] = MPlug(node, t0Attr).asFloat();
	thickness[1] = MPlug(node, t1Attr).asFloat();
	thickness[2] = MPlug(node, t2Attr).asFloat();
	thickness[3] = MPlug(node, t3Attr).asFloat();
	thickness[4] = MPlug(node, t4Attr).asFloat();
	FeatherGeomParam * param = featherGeomParameter();
	param->setCovert(i, nps, chs, thickness);
}

void MAvianArm::setFeatherOrientationParam(const MObject & node,
					const MObject & m0Attr,
					const MObject & m1Attr,
					const MObject & m2Attr,
					const MObject & u0rzAttr)
{
	MMatrix inboardM = AHelper::getMatrixAttr(node, m0Attr);
	AHelper::ConvertToMatrix44F(*inboardMarixR(), inboardM);
	MMatrix midsect1M = AHelper::getMatrixAttr(node, m1Attr);
	AHelper::ConvertToMatrix44F(*midsection0MarixR(), midsect1M);
	MMatrix midsect2M = AHelper::getMatrixAttr(node, m2Attr);
	AHelper::ConvertToMatrix44F(*midsection1MarixR(), midsect2M);
	
	Matrix33F invrot = invPrincipleMatrixR()->rotation();
	Matrix33F orient[4];
	orient[0] = inboardMarixR()->rotation();
	orient[0] = invrot * orient[0];
	
	orient[1] = midsection0MarixR()->rotation();
	orient[2] = midsection1MarixR()->rotation();
	orient[3] = secondDigitMatirxR()->rotation();
/// offset second digit 
	Quaternion q(0.1f, Vector3F::YAxis);
	Matrix33F offset(q);
	orient[3] *= offset;
	orient[3] = invrot * orient[3];
	
	float covertRz[4];
	memset(covertRz, 0, 4*4);
	covertRz[0] = MPlug(node, u0rzAttr).asFloat() * .1f;
	
	FeatherOrientationParam * param = orientationParameter();
	param->set(orient, covertRz);
}

void MAvianArm::setFeatherDeformationParam(const MObject & node, 
					const MObject & brt0Attr, 
					const MObject & brt1Attr, 
					const MObject & brt2Attr, 
					const MObject & brt3Attr)
{
	MMatrix brtM;
	Matrix33F orient[4];
	
	brtM = AHelper::getMatrixAttr(node, brt0Attr);
	AHelper::ConvertToMatrix33F(orient[0], brtM);
	
	brtM = AHelper::getMatrixAttr(node, brt1Attr);
	AHelper::ConvertToMatrix33F(orient[1], brtM);
	
	brtM = AHelper::getMatrixAttr(node, brt2Attr);
	AHelper::ConvertToMatrix33F(orient[2], brtM);
	
	brtM = AHelper::getMatrixAttr(node, brt3Attr);
	AHelper::ConvertToMatrix33F(orient[3], brtM);
	
	FeatherDeformParam * param = featherDeformParameter();
	param->set(orient);
	
}

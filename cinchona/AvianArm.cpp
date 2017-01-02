/*
 *  AvianArm.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "AvianArm.h"
#include "Ligament.h"
#include <AllMath.h>

using namespace aphid;

AvianArm::AvianArm()
{
	m_skeletonMatrices = new Matrix44F[7];
	m_leadingLigament = new Ligament(3);
	m_trailingLigament = new Ligament(4);
	m_secondDigitLength = 2.f;
}

AvianArm::~AvianArm()
{
	delete[] m_skeletonMatrices;
	delete m_leadingLigament;
	delete m_trailingLigament;
}

void AvianArm::set2ndDigitLength(const float & x)
{ m_secondDigitLength = x; }

const Matrix44F & AvianArm::skeletonMatrix(const int & idx) const
{ return m_skeletonMatrices[idx]; }

Matrix44F * AvianArm::skeletonMatricesR()
{ return m_skeletonMatrices; }

Matrix44F * AvianArm::principleMatricesR()
{ return &m_skeletonMatrices[5]; }

Matrix44F * AvianArm::invPrincipleMatricesR()
{ return &m_skeletonMatrices[6]; }

Vector3F AvianArm::shoulderPosition() const
{ return m_skeletonMatrices[0].getTranslation(); }

Vector3F AvianArm::elbowPosition() const
{ return m_skeletonMatrices[1].getTranslation(); }

Vector3F AvianArm::wristPosition() const
{ return m_skeletonMatrices[3].getTranslation(); }

Vector3F AvianArm::secondDigitPosition() const
{ return m_skeletonMatrices[4].getTranslation(); }

Vector3F AvianArm::secondDigitEndPosition() const
{ return (secondDigitPosition() + m_skeletonMatrices[4].getSide() * m_secondDigitLength ); }

const Ligament & AvianArm::leadingLigament() const
{ return *m_leadingLigament; }
	
const Ligament & AvianArm::trailingLigament() const
{ return *m_trailingLigament; }
	
	
bool AvianArm::updatePrincipleMatrix()
{ 
	Vector3F side = wristPosition() - shoulderPosition();
	if(side.length2() < 1.0e-4f) {
		std::cout<<"\n ERROR AvianArm updatePrincipleMatrix wrist too close to shoulder";
		return false;
	}
	
	side.normalize();
	Vector3F up = m_skeletonMatrices[0].getUp();
	Vector3F front = side.cross(up);
	front.normalize();
	
	principleMatricesR()->setOrientations(side, up, front );
	principleMatricesR()->setTranslation(shoulderPosition() );
	*invPrincipleMatricesR() = *principleMatricesR();
	invPrincipleMatricesR()->inverse();
	
	return true;
}

void AvianArm::updateLigaments()
{
	const Vector3F shoulderP = shoulderPosition();
	const Vector3F elbowP = elbowPosition();
	const Vector3F wristP = wristPosition();
	const Vector3F snddigitP = secondDigitPosition();
	const Vector3F endP = secondDigitEndPosition();
	
	m_leadingLigament->setKnotPoint(0, shoulderP);
	m_leadingLigament->setKnotPoint(1, wristP);
	m_leadingLigament->setKnotPoint(2, snddigitP);
	m_leadingLigament->setKnotPoint(3, endP);
	
	m_leadingLigament->update();
	
	m_trailingLigament->setKnotPoint(0, shoulderP);
	m_trailingLigament->setKnotPoint(1, elbowP);
	m_trailingLigament->setKnotPoint(2, wristP);
	m_trailingLigament->setKnotPoint(3, snddigitP);
	m_trailingLigament->setKnotPoint(4, endP);
	
	m_trailingLigament->update();
}

void AvianArm::setLeadingLigamentOffset(const int & idx,
							const Vector3F & v) const
{
	m_leadingLigament->setKnotOffset(idx, v);
}
	
void AvianArm::setTrailingLigamentOffset(const int & idx,
							const Vector3F & v) const
{
	m_trailingLigament->setKnotOffset(idx, v);
}

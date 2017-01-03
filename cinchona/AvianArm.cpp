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
#include "FeatherMesh.h"
#include "FeatherObject.h"
#include "FeatherGeomParam.h"
#include <AllMath.h>

using namespace aphid;

AvianArm::AvianArm()
{
	m_featherGeomParam = new FeatherGeomParam;
	m_skeletonMatrices = new Matrix44F[11];
	m_leadingLigament = new Ligament(3);
	m_trailingLigament = new Ligament(4);
	m_secondDigitLength = 2.f;
}

AvianArm::~AvianArm()
{
	delete m_featherGeomParam;
	delete[] m_skeletonMatrices;
	delete m_leadingLigament;
	delete m_trailingLigament;
	clearFeathers();
}

void AvianArm::set2ndDigitLength(const float & x)
{ m_secondDigitLength = x; }

const Matrix44F & AvianArm::skeletonMatrix(const int & idx) const
{ return m_skeletonMatrices[idx]; }

Matrix44F * AvianArm::skeletonMatricesR()
{ return m_skeletonMatrices; }

Matrix44F * AvianArm::principleMatrixR()
{ return &m_skeletonMatrices[5]; }

Matrix44F * AvianArm::invPrincipleMatrixR()
{ return &m_skeletonMatrices[6]; }

Matrix44F * AvianArm::secondDigitMatirxR()
{ return &m_skeletonMatrices[4]; }

Matrix44F * AvianArm::handMatrixR()
{ return &m_skeletonMatrices[7]; }
	
Matrix44F * AvianArm::invHandMatrixR()
{ return &m_skeletonMatrices[8]; }

Matrix44F * AvianArm::fingerMatrixR()
{ return &m_skeletonMatrices[9]; }

Matrix44F * AvianArm::invFingerMatrixR()
{ return &m_skeletonMatrices[10]; }

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
	
	principleMatrixR()->setOrientations(side, up, front );
	principleMatrixR()->setTranslation(shoulderPosition() );
	*invPrincipleMatrixR() = *principleMatrixR();
	invPrincipleMatrixR()->inverse();
	
	return true;
}

bool AvianArm::updateHandMatrix()
{
	Vector3F side = secondDigitPosition() - m_skeletonMatrices[2].getTranslation();
	if(side.length2() < 1.0e-4f) {
		std::cout<<"\n ERROR AvianArm updateHandMatrix 2nd_digit too close to radius";
		return false;
	}
	
	side = invPrincipleMatrixR()->transformAsNormal(side);
	side.normalize();
	
	Vector3F up = skeletonMatricesR()[2].getUp();
	up = invPrincipleMatrixR()->transformAsNormal(up);
	up.normalize();
	
	Vector3F front = side.cross(up);
	front.normalize();
	
	handMatrixR()->setOrientations(side, up, front );
	handMatrixR()->setTranslation(wristPosition() );
	*invHandMatrixR() = *handMatrixR();
	invHandMatrixR()->inverse();
	
	return true;
}

bool AvianArm::updateFingerMatrix()
{
	Vector3F side = secondDigitEndPosition() - wristPosition();
	if(side.length2() < 1.0e-4f) {
		std::cout<<"\n ERROR AvianArm updateHandMatrix 2nd_digit_end too close to wrist";
		return false;
	}
	
	side = invPrincipleMatrixR()->transformAsNormal(side);
	side.normalize();
	
	Vector3F up = handMatrixR()->getUp();
	up.normalize();
	
	Vector3F front = side.cross(up);
	front.normalize();
	
	fingerMatrixR()->setOrientations(side, up, front );
	fingerMatrixR()->setTranslation(secondDigitPosition() );
	*invFingerMatrixR() = *fingerMatrixR();
	invFingerMatrixR()->inverse();

	return true;
}

void AvianArm::updateLigaments()
{
	Vector3F elbowP = elbowPosition();
	elbowP = invPrincipleMatrixR()->transform(elbowP);
	Vector3F wristP = wristPosition();
	wristP = invPrincipleMatrixR()->transform(wristP);
	Vector3F snddigitP = secondDigitPosition();
	snddigitP = invPrincipleMatrixR()->transform(snddigitP);
	Vector3F endP = secondDigitEndPosition();
	endP = invPrincipleMatrixR()->transform(endP);
	
	m_leadingLigament->setKnotPoint(0, Vector3F::Zero);
	m_leadingLigament->setKnotPoint(1, wristP);
	m_leadingLigament->setKnotPoint(2, snddigitP);
	m_leadingLigament->setKnotPoint(3, endP);
	
	m_leadingLigament->update();
	
	m_trailingLigament->setKnotPoint(0, Vector3F::Zero);
	m_trailingLigament->setKnotPoint(1, elbowP);
	m_trailingLigament->setKnotPoint(2, wristP);
	m_trailingLigament->setKnotPoint(3, snddigitP);
	m_trailingLigament->setKnotPoint(4, endP);
	
	m_trailingLigament->update();
}

void AvianArm::setLeadingLigamentOffset(const int & idx,
							const Vector3F & v)
{
	m_leadingLigament->setKnotOffset(idx, v);
}
	
void AvianArm::setTrailingLigamentOffset(const int & idx,
							const Vector3F & v)
{
	m_trailingLigament->setKnotOffset(idx, v);
}

void AvianArm::setLeadingLigamentTangent(const int & idx,
							const aphid::Vector3F & v)
{
	m_leadingLigament->setKnotTangent(idx, v);
}

void AvianArm::setTrailingLigamentTangent(const int & idx,
							const aphid::Vector3F & v)
{
	m_trailingLigament->setKnotTangent(idx, v);
}

Ligament * AvianArm::leadingLigamentR()
{ return m_leadingLigament; }

FeatherGeomParam * AvianArm::featherGeomParameter()
{ return m_featherGeomParam; }

bool AvianArm::isFeatherGeomParameterChanged() const
{ return m_featherGeomParam->isChanged(); }

int AvianArm::numFeathers() const
{ return m_feathers.size(); }

const FeatherObject * AvianArm::feather(int i) const
{ return m_feathers[i]; }

void AvianArm::clearFeathers()
{
    std::vector<FeatherObject *>::iterator it = m_feathers.begin();
    for(;it!= m_feathers.end();++it) {
        delete *it;
    }
    m_feathers.clear();
}

void AvianArm::updateFeatherGeom()
{
	if(!isFeatherGeomParameterChanged() ) {
		return;
	}
	
	clearFeathers();
	
	const int nseg = m_featherGeomParam->numSegments();
	for(int i=0;i<nseg;++i) {
	    const int & nf = m_featherGeomParam->numFeatherOnSegment(i);
	    const float * xs = m_featherGeomParam->xOnSegment(i);
	    for(int j=0;j<nf;++j) {
	        FeatherMesh * msh = new FeatherMesh(20.f, 0.02f, 0.4f, 0.15f);
	        msh->create(20, 2);
	        FeatherObject * f = new FeatherObject(msh);
	        Vector3F p = m_trailingLigament->getPoint(i, xs[j] );
	        f->setTranslation(p);
	        
	        m_feathers.push_back(f);
	    }
	}
	
	std::cout<<"AvianArm update n feather geom "<<numFeathers();
	std::cout.flush();
	
}

void AvianArm::updateFeatherTransform()
{
    int it = 0;
    const int nseg = m_featherGeomParam->numSegments();
	for(int i=0;i<nseg;++i) {
	    const int & nf = m_featherGeomParam->numFeatherOnSegment(i);
	    const float * xs = m_featherGeomParam->xOnSegment(i);
	    for(int j=0;j<nf;++j) {
	        FeatherObject * f = m_feathers[it];
	        Vector3F p = m_trailingLigament->getPoint(i, xs[j] );
	        f->setTranslation(p);
	        
	        it++;
	    }
	}
}

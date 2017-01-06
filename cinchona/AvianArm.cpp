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
#include "FeatherDeformParam.h"
#include "WingRib.h"
#include "WingSpar.h"
#include <AllMath.h>
#include <math/linspace.h>

using namespace aphid;

AvianArm::AvianArm()
{
	m_orientationParam = new FeatherOrientationParam;
	m_featherGeomParam = new FeatherGeomParam;
	m_featherDeformParam = new FeatherDeformParam;
	m_skeletonMatrices = new Matrix44F[NUM_MAT];
	m_leadingLigament = new Ligament(3);
	m_trailingLigament = new Ligament(3);
	m_secondDigitLength = 2.f;
	m_featherX = NULL;
	for(int i=0;i<5;++i) {
		m_ribs[i] = new WingRib(2.f, 0.f, .5f, .2f);
	}
	for(int i=0;i<4;++i) {
		m_spars[i] = new WingSpar(4);
	}
}

AvianArm::~AvianArm()
{
	delete m_orientationParam;
	delete m_featherGeomParam;
	delete m_featherDeformParam;
	delete[] m_skeletonMatrices;
	delete m_leadingLigament;
	delete m_trailingLigament;
	if(m_featherX) {
		delete[] m_featherX;
	}
	clearFeathers();
	for(int i=0;i<5;++i) {
		delete m_ribs[i];
	}
	for(int i=0;i<4;++i) {
		delete m_spars[i];
	}
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

Matrix44F * AvianArm::inboardMarixR()
{ return &m_skeletonMatrices[11]; }

Matrix44F * AvianArm::midsection0MarixR()
{ return &m_skeletonMatrices[12]; }

Matrix44F * AvianArm::midsection1MarixR()
{ return &m_skeletonMatrices[13]; }

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
	Vector3F side = secondDigitMatirxR()->getSide();
	
	//side = invPrincipleMatrixR()->transformAsNormal(side);
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
	m_trailingLigament->setKnotPoint(3, endP);
	
	m_trailingLigament->update();
	
	updateRibs();
	updateSpars();
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
	
	int it = 0;
	for(int i=0;i<nseg;++i) {
	    const int nf = m_featherGeomParam->numFeatherOnSegment(i);
	    it += nf-1;
	}
	
	const float dx = 1.f / (float)it;
	
	it = 0;
	for(int i=0;i<nseg;++i) {
	    const int nf = m_featherGeomParam->numFeatherOnSegment(i);
	    const float * xs = m_featherGeomParam->xOnSegment(i);
	    for(int j=1;j<nf;++j) {
			float vx = dx * (it + 0.5f);
			it++;
			const float c = m_featherGeomParam->predictChord(&vx);
			const float t = m_featherGeomParam->predictThickness(&vx);
			
	        FeatherMesh * msh = new FeatherMesh(c, 0.03f, 0.14f, t);
	        msh->create(20, 2);
	        FeatherObject * f = new FeatherObject(msh);
	        Vector3F p = m_trailingLigament->getPoint(i, xs[j] );
	        f->setTranslation(p);
	        
	        m_feathers.push_back(f);
	    }
	}
	
	const int n = numFeathers();
	std::cout<<"AvianArm update n feather geom "<<n;
	std::cout.flush();
	
	if(m_featherX) {
		delete[] m_featherX;
	}
	
	m_featherX = new float[n];
/// from tip 0 to root 1
	linspace_center_reverse<float>(m_featherX, 0.f, 1.f, n);
	
}

FeatherOrientationParam * AvianArm::orientationParameter()
{ return m_orientationParam; }

void AvianArm::updateFeatherTransform()
{
/// point on ligament
    int it = 0;
    const int nseg = m_featherGeomParam->numSegments();
	for(int i=0;i<nseg;++i) {
	    const int nf = m_featherGeomParam->numFeatherOnSegment(i);
	    const float * xs = m_featherGeomParam->xOnSegment(i);
	    for(int j=1;j<nf;++j) {
	        FeatherObject * f = m_feathers[it];
	        Vector3F p = m_trailingLigament->getPoint(i, xs[j] );
	        f->setTranslation(p);
	        
	        it++;
	    }
	}
/// interpolate orientation
	FeatherOrientationParam * param = orientationParameter();
	if(!param->isChanged()
		&& !isFeatherGeomParameterChanged() ) {
		return;
	}
	
	Matrix33F rotM;
	const int n = numFeathers();
	for(int i=0;i<n;++i) {
		param->predictRotation(rotM, &m_featherX[i]);
		m_feathers[i]->setRotation(rotM);
	}
	
}

FeatherDeformParam * AvianArm::featherDeformParameter()
{ return m_featherDeformParam; }

void AvianArm::updateFeatherDeformation()
{
	FeatherDeformParam * param = featherDeformParameter();
	if(!param->isChanged()
	 && !isFeatherGeomParameterChanged() ) {
		return;
	}
	
	const float & longestC = featherGeomParameter()->longestChord();
	
	Matrix33F deformM;
	const int n = numFeathers();
	for(int i=0;i<n;++i) {
		float relspeed = m_feathers[i]->mesh()->chord() / longestC * 0.1f;
		
		param->predictRotation(deformM, &m_featherX[i], relspeed);
		
		m_feathers[i]->deform(deformM);
	}

}

static const int sLeadRibSeg[5] = {
0, 0, 1, 2, 2
};

static const int sTrailRibSeg[5] = {
0, 1, 1, 2, 2
};

static const float sLeadRibX[5] = {
0.05f, 0.5f, 0.01f, 0.1f, 0.73f
};

static const float sTrailRibX[5] = {
0.1f, 0.01f, .9f, 0.41f, 0.83f
};

void AvianArm::updateRibs()
{
	float c;
	Vector3F p, q, side, up, front;
	
	for(int i=0;i<5;++i) {
		p = leadingLigament().getPoint(sLeadRibSeg[i], sLeadRibX[i]);
		q = trailingLigament().getPoint(sTrailRibSeg[i], sTrailRibX[i]);
		side = q - p;
		c = p.distanceTo(q);
		side /= c;
		m_ribs[i]->setCMPT(c, 0.f, 0.4f, 0.2f);
		m_ribs[i]->setTranslation(p);
		front = side.cross(Vector3F::YAxis);
		front.normalize();
		
		up = front.cross(side);
		
		m_ribs[i]->setOrientations(side, up, front);
		
		q = trailingLigament().getDerivative(sTrailRibSeg[i], sTrailRibX[i]);
		m_ribs[i]->setSparTangent(q);
		
	}
}

static const float sSparX[4] = {
0.8f, 0.6f,
-0.2f,-0.4f
};

void AvianArm::updateSpars()
{
	Vector3F pnt, tng;
	for(int i=0;i<4;++i) {
		WingSpar & spari = *m_spars[i];
		for(int j=0;j<5;++j) {
			tng = m_ribs[j]->sparTangent();
			m_ribs[j]->getPoint(pnt, sSparX[i]);
			spari.setKnot(j, pnt, tng);
		}
	}
}

const WingRib * AvianArm::rib(int i) const
{ return m_ribs[i]; }

const WingSpar * AvianArm::spar(int i) const
{ return m_spars[i]; }

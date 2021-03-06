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
#include "Geom1LineParam.h"
#include "FeatherDeformParam.h"
#include "WingRib.h"
#include "WingSpar.h"
#include <math/PseudoNoise.h>
#include <AllMath.h>

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

	for(int i=0;i<5;++i) {
/// thickness of rib
		m_ribs[i] = new WingRib(2.f, 0.001f, .3f, .33f);
	}
	for(int i=0;i<4;++i) {
		m_spars[i] = new WingSpar(4);
	}
	m_starboard = false;
}

AvianArm::~AvianArm()
{
	delete m_orientationParam;
	delete m_featherGeomParam;
	delete m_featherDeformParam;
	delete[] m_skeletonMatrices;
	delete m_leadingLigament;
	delete m_trailingLigament;
	
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

Matrix44F * AvianArm::radiusMatrixR()
{ return &m_skeletonMatrices[2]; }

Vector3F AvianArm::shoulderPosition() const
{ return m_skeletonMatrices[0].getTranslation(); }

Vector3F AvianArm::elbowPosition() const
{ return m_skeletonMatrices[1].getTranslation(); }

Vector3F AvianArm::wristPosition() const
{ return m_skeletonMatrices[3].getTranslation(); }

Vector3F AvianArm::secondDigitPosition() const
{ return m_skeletonMatrices[4].getTranslation(); }

Vector3F AvianArm::secondDigitEndPosition() const
{ 
	Vector3F p = secondDigitPosition() + m_skeletonMatrices[4].getSide() * m_secondDigitLength;
	return p; 
}

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
	Vector3F side = secondDigitPosition() - radiusMatrixR()->getTranslation();
	if(side.length2() < 1.0e-4f) {
		std::cout<<"\n ERROR AvianArm updateHandMatrix 2nd_digit too close to radius";
		return false;
	}
	
	//side = invPrincipleMatrixR()->transformAsNormal(side);
	side.normalize();
	
	Vector3F up = radiusMatrixR()->getUp();
	
	//up = invPrincipleMatrixR()->transformAsNormal(up);
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
	
	updateFeatherLineGeom(m_featherGeomParam->line(0), 
							m_trailingLigament->curve() );
	
	const float maxC = m_featherGeomParam->longestChord();
	
	for(int i=1;i<5;++i) {
		updateFeatherLineGeom(m_featherGeomParam->line(i),
							spar(i-1), maxC );
	}
	
	const int n = numFeathers();
	std::cout<<"\n AvianArm update n feather geom "<<n
			<<"\n longest chord "<<maxC;
	std::cout.flush();
		
}

FeatherOrientationParam * AvianArm::orientationParameter()
{ return m_orientationParam; }

bool AvianArm::isFeatherOrientationChanged() const
{ return m_orientationParam->isChanged(); }

void AvianArm::updateFeatherTransform()
{
	const float * yawwei = orientationParameter()->yawNoise();
	int it = 0;
	for(int i=0;i<5;++i) {
		updateFeatherLineRotationOffset(m_featherGeomParam->line(i), 
			yawwei, it);
	}
	
	it = 0;
	updateFeatherLineTranslation(m_featherGeomParam->line(0), 
		m_trailingLigament, it);
	
    for(int i=1;i<5;++i) {
		updateFeatherLineTranslation(m_featherGeomParam->line(i), 
			spar(i-1), it);
	}
	
	it = 0;
	for(int i=0;i<5;++i) {
		updateFeatherLineRotation(i, m_featherGeomParam->line(i), it);
	}
	
	it = 0;
	for(int i=0;i<5;++i) {
		updateWarp(m_featherGeomParam->line(i), it);
	}
}

FeatherDeformParam * AvianArm::featherDeformParameter()
{ return m_featherDeformParam; }

void AvianArm::updateFeatherDeformation()
{
	FeatherDeformParam * param = featherDeformParameter();
	if(!param->isChanged()
	 && !isFeatherGeomParameterChanged() 
	 && !isFeatherOrientationChanged() ) {
		return;
	}
	
	const float & longestC = featherGeomParameter()->longestChord();
	
	Matrix33F deformM;
	const int n = numFeathers();
	for(int i=0;i<n;++i) {
		const FeatherMesh * msh = m_feathers[i]->mesh();
		float relspeed = msh->chord() / longestC * 0.076f * 31.f / (float)msh->numVertexRows();
		
		param->predictRotation(deformM, m_feathers[i]->predictX(), relspeed);
		
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
0.05f, 0.5f, 0.01f, 0.1f, 0.95f
};

static const float sTrailRibX[5] = {
0.1f, 0.01f, .9f, 0.41f, 0.9f
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
		
		if(i<2) {
			up = Vector3F::YAxis;
		} else {
			up = handMatrixR()->getUp();
			up = invPrincipleMatrixR()->transformAsNormal(up);
		}
		
		front = side.cross(up);
		front.normalize();
		
		up = front.cross(side);
		
		m_ribs[i]->setOrientations(side, up, front);
		
		q = trailingLigament().getDerivative(sTrailRibSeg[i], sTrailRibX[i]);
		m_ribs[i]->setSparTangent(q);
		
	}
}

static const float sSparX[8] = {
0.65f, 0.5f,
-0.35f,-0.5f,
-0.35f,-0.5f,
0.65f, 0.5f
};

void AvianArm::updateSpars()
{
	const float * usesparx = &sSparX[0];
	//if(isStarboard()) {
	//	usesparx = &sSparX[4];
	//}
	
	Vector3F pnt, tng;
	for(int i=0;i<4;++i) {
		WingSpar & spari = *m_spars[i];
		for(int j=0;j<5;++j) {
			tng = m_ribs[j]->sparTangent();
			m_ribs[j]->getPoint(pnt, usesparx[i]);
			spari.setKnot(j, pnt, tng);
		}
	}
}

const WingRib * AvianArm::rib(int i) const
{ return m_ribs[i]; }

const WingSpar * AvianArm::spar(int i) const
{ return m_spars[i]; }

void AvianArm::updateFeatherLineGeom(Geom1LineParam * line,
				const HermiteInterpolatePiecewise<float, Vector3F > * curve,
				const float & maxC)
{
	const int nseg = line->numSegments();
	const int ngeom = line->numGeoms();
	
	float * vxs = new float[ngeom];
	line->calculateX<HermiteInterpolatePiecewise<float, Vector3F > >(vxs, curve);
	
	int gx = 32;
	
	int it = 0;
	for(int i=0;i<nseg;++i) {
	    const int nf = line->numFeatherOnSegment(i);
	    for(int j=1;j<nf;++j) {
			const float & vx = vxs[it+j];
			
			
			const float c = line->predictChord(&vx);
			const float t = line->predictThickness(&vx);
			
	        FeatherMesh * msh = new FeatherMesh(c, 0.03f, 0.14f, t);
			
			if(maxC < c) {
				gx = 32;
			} else {
				gx = 32.f * (c / maxC);
				if(gx < 5) {
					gx = 5;
				}
			}
			
	        msh->create(gx, 2);
	        
			FeatherObject * f = new FeatherObject(msh);
			f->setPredictX(vx);
			
			if(isStarboard()) {
				f->flipZ();
			}

	        m_feathers.push_back(f);
	    }
		it += nf;
	}
	
	delete[] vxs;
}

void AvianArm::updateFeatherLineTranslation(Geom1LineParam * line, 
					const Ligament * lig,
					int & it)
{
	const int nseg = line->numSegments();
	for(int i=0;i<nseg;++i) {
	    const int nf = line->numFeatherOnSegment(i);
	    const float * xs = line->xOnSegment(i);
	    for(int j=1;j<nf;++j) {
	        FeatherObject * f = m_feathers[it];
/// point on ligament
	        Vector3F p = lig->getPoint(i, xs[j] );
	        f->setTranslation(p);
	        
	        it++;
	    }
	}
}

void AvianArm::updateFeatherLineTranslation(Geom1LineParam * line, 
							const WingSpar * spr,
							int & it)
{
	const int nseg = line->numSegments();
	for(int i=0;i<nseg;++i) {
	    const int nf = line->numFeatherOnSegment(i);
	    const float * xs = line->xOnSegment(i);
	    for(int j=1;j<nf;++j) {
	        FeatherObject * f = m_feathers[it];
/// point on spar
	        Vector3F p = spr->getPoint(i, xs[j] );
	        f->setTranslation(p);
	        
	        it++;
	    }
	}
}
	
void AvianArm::updateFeatherLineRotation(const int & iline,
							Geom1LineParam * line, 
							int & it)
{	
	FeatherOrientationParam * param = orientationParameter();
	if(!param->isChanged()
		&& !isFeatherGeomParameterChanged() ) {
		return;
	}
	
	const int n = line->numGeomsM1();
	
/// interpolate orientation	
	Matrix33F rotM;
	for(int i=0;i<n;++i) {
		FeatherObject * fea = m_feathers[it];
		param->predictLineRotation(rotM, iline, fea->predictX() );
		fea->setRotation(rotM);
		it++;
	}
}

void AvianArm::updateWarp(Geom1LineParam * line, 
							int & it)
{
	const int n = line->numGeomsM1();
/// current
	Vector3F samp[2];
/// from inboard
	Vector3F dev0[2];
/// to outboard
	Vector3F dev1[2];
	
	for(int i=0;i<n;++i) {
		FeatherObject * f = m_feathers[it];
		f->getEndPoints(samp);
			
		if(i==0) {
			feather(it+1)->getEndPoints(dev1);
			dev1[0] -= samp[0];
			dev1[1] -= samp[1];
			dev0[0] = dev1[0];
			dev0[1] = dev1[1];
		} else if(i+1 == n) {
			feather(it-1)->getEndPoints(dev0);
			dev0[0] = samp[0] - dev0[0];
			dev0[1] = samp[1] - dev0[1];
			//dev1[0] = dev0[0];
			//dev1[1] = dev0[1];
		} else {
			feather(it-1)->getEndPoints(dev0);
			//feather(it+1)->getEndPoints(dev1);
			dev0[0] = samp[0] - dev0[0];
			dev0[1] = samp[1] - dev0[1];
			//dev1[0] -= samp[0];
			//dev1[1] -= samp[1];
		}
		
		f->setWarp(dev0, isStarboard() );
		
		it++;
	}
}

void AvianArm::updateFeatherLineRotationOffset(Geom1LineParam * line, 
							const float * yawWeight,
							int & it)
{
	const float maxC = featherGeomParameter()->longestChord();
	
	PseudoNoise psn;
	
	float vpitch, lastPitch = 0.f;

	float vyaw = 0.f;
	const int n = line->numGeomsM1();
/// gap between 
	float delta = .0053f;
	
	//if(isStarboard()) {
	//	delta *= -1.f;
	//}

	for(int i=n-1;i>=0;--i) {
		FeatherObject * f = m_feathers[it+i];
		
		if(yawWeight[0] > 0.f) {
			vyaw = (psn.rfloat(it+i) - .5f) * yawWeight[0] * (1.f - f->mesh()->chord() / maxC);
		}
		
/// rotate down from root to tip
		vpitch = lastPitch + delta;
		f->setRotationOffset(0.f, vyaw, vpitch );
			
		lastPitch = vpitch;
	}
	it+=n;
}

void AvianArm::setStarboard(bool x)
{ m_starboard = x; }

const bool & AvianArm::isStarboard() const
{ return m_starboard; }

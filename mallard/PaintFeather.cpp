/*
 *  PaintFeather.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "PaintFeather.h"
#include "MlCalamus.h"
#include "CalamusSkin.h"
#include <FloodCondition.h>
PaintFeather::PaintFeather(CalamusSkin * skin, std::deque<unsigned> * indices, FloodCondition * density) 
{
	m_skin = skin;
	m_indices = indices;
	m_density = density;
}

PaintFeather::~PaintFeather() {}

void PaintFeather::computeWeights(const Vector3F & center, const float & radius)
{
	const unsigned num = m_indices->size();
	if(num < 1) return;
	
	m_weights.reset(new float[num]);
	float drop;
	Vector3F p;
	for(unsigned i =0; i < num; i++) {
		MlCalamus * c = m_skin->getCalamus(m_indices->at(i));
		
		m_skin->getPointOnBody(c, p);
		drop = p.distanceTo(center) / radius;
		drop = 1.f - drop * drop;

		m_weights[i] = drop;
    }
}

void PaintFeather::perform(PaintMode mode, const Vector3F & brushInput)
{
	if(brushInput.length() < 10e-4) return;
	switch (mode) {
		case MDirection:
			brushDirection(brushInput);
			break;
		case MLength:
			brushLength(brushInput);
			break;
		case MRoll:
			brushRoll(brushInput);
			break;
		default:
			break;
	}
}

void PaintFeather::brushDirection(const Vector3F & brushInput)
{
	const unsigned num = m_indices->size();
	if(num < 1) return;
	
	Matrix33F space, rotfrm;
	Vector3F div, zdir;
	float rotX;
	unsigned i;
	
	for(i =0; i < num; i++) {
		MlCalamus * c = m_skin->getCalamus(m_indices->at(i));
		
		m_skin->tangentSpace(c, space);		
		space.inverse();
		
		div = space.transform(brushInput);
		div.x = 0.f;
		
		zdir.set(0.f, 0.f, 1.f);
		zdir.rotateAroundAxis(Vector3F::XAxis, c->rotateX());
		zdir += div * m_weights[i] * .5f;
		
		rotX = zdir.angleX();

		c->setRotateX(rotX);
    }
}

void PaintFeather::brushLength(const Vector3F & brushInput)
{
	const unsigned num = m_indices->size();
	if(num < 1) return;
	
	Matrix33F space;
	Vector3F zdir;
	unsigned i;
	float dscale;
	
	float activeMeanScale = 0.f;
	Vector3F activeMeanDir;
	boost::scoped_array<float> densityScales(new float[num]);
	for(i =0; i < num; i++) {
		MlCalamus * c = m_skin->getCalamus(m_indices->at(i));
		
		m_skin->tangentSpace(c, space);
		
		dscale = 1.f;
		m_density->reduceScale(c->faceIdx(), c->patchU(), c->patchV(), dscale);
		
		densityScales[i] = dscale;
		
		zdir.set(0.f, 0.f, 1.f);
		zdir.rotateAroundAxis(Vector3F::XAxis, c->rotateX());
		zdir = space.transform(zdir);
		activeMeanDir += zdir;
		activeMeanScale += c->realScale() / dscale;
	}
	activeMeanScale /= num;
	activeMeanDir /= (float)num;

	if(brushInput.dot(activeMeanDir) < 0.f) activeMeanScale *= .9f;
	else activeMeanScale *= 1.1f;
	
	float wei;
	for(i =0; i < num; i++) {
		MlCalamus * c = m_skin->getCalamus(m_indices->at(i));

		wei = m_weights[i];

		c->scaleLength(activeMeanScale * densityScales[i] * wei + c->realScale() * (1.f - wei));
    }
}

void PaintFeather::brushRoll(const Vector3F & brushInput)
{
	const unsigned num = m_indices->size();
	if(num < 1) return;
	
	Matrix33F space;
	Vector3F zdir;
	unsigned i;
	
	float activeMeanPitch = 0.f;
	Vector3F activeMeanDir;
	for(i =0; i < num; i++) {
		MlCalamus * c = m_skin->getCalamus(m_indices->at(i));
		activeMeanPitch += c->rotateY();
		
		m_skin->tangentSpace(c, space);
		zdir.set(0.f, 0.f, 1.f);
		zdir.rotateAroundAxis(Vector3F::XAxis, c->rotateX());
		zdir = space.transform(zdir);
		activeMeanDir += zdir;
	}
	activeMeanPitch /= num;
	activeMeanDir /= (float)num;

	if(brushInput.dot(activeMeanDir) < 0.f) activeMeanPitch -= .1f;
	else activeMeanPitch += .1f;
	
	float wei;
	for(i =0; i < num; i++) {
		MlCalamus * c = m_skin->getCalamus(m_indices->at(i));

		wei = m_weights[i];
		c->setRotateY(activeMeanPitch * wei + c->rotateY() * (1.f - wei));
    }
}

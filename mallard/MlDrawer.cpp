/*
 *  MlDrawer.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlDrawer.h"
#include "MlCalamus.h"
#include "MlFeather.h"
#include "MlSkin.h"
#include "MlTessellate.h"
#include <AccPatchMesh.h>
#include <PointInsidePolygonTest.h>
#include <HBase.h>
#include <sstream>
MlDrawer::MlDrawer()
{
	std::cout<<"Feather buffer ";
	m_featherTess = new MlTessellate;
	m_currentFrame = -9999;
}

MlDrawer::~MlDrawer() 
{
	delete m_featherTess;
}

void MlDrawer::draw(MlSkin * skin) const
{
	if(skin->numFeathers() > 0) drawBuffer();
}

void MlDrawer::hideAFeather(MlCalamus * c)
{
	const unsigned loc = c->bufferStart();
	
	setIndex(loc);
	
	m_featherTess->setFeather(c->feather());
	
	unsigned i;
	const unsigned nvpf = m_featherTess->numIndices();
	for(i = 0; i < nvpf; i++) {
		vertices()[0] = 0.f;
		vertices()[1] = 0.f;
		vertices()[2] = 0.f;
		next();
	}
}

void MlDrawer::hideActive(MlSkin * skin)
{
	const unsigned num = skin->numActive();
	if(num < 1) return;
	
	unsigned i;
	for(i = 0; i < num; i++) {
		MlCalamus * c = skin->getActive(i);
		hideAFeather(c);
	}
}

void MlDrawer::updateActive(MlSkin * skin)
{
	const unsigned num = skin->numActive();
	if(num < 1) return;
	
	unsigned i;
	for(i = 0; i < num; i++) {
		MlCalamus * c = skin->getActive(i);
		computeFeather(skin, c);
		updateBuffer(c);
	}
}

void MlDrawer::updateBuffer(MlCalamus * c)
{
	tessellate(c->feather());
	
	const unsigned nvpf = m_featherTess->numIndices();
	const unsigned startv = c->bufferStart();
	setIndex(startv);
	
	unsigned i, j;
	Vector3F v;
	Vector2F st;
	for(i = 0; i < nvpf; i++) {
		j = m_featherTess->indices()[i];
		v = m_featherTess->vertices()[j];
		vertices()[0] = v.x;
		vertices()[1] = v.y;
		vertices()[2] = v.z;
		
		v = m_featherTess->normals()[j];
		normals()[0] = v.x;
		normals()[1] = v.y;
		normals()[2] = v.z;
		
		st = m_featherTess->texcoords()[j];
		texcoords()[0] = st.x;
		texcoords()[1] = st.y;
		
		next();
	}
}

void MlDrawer::addToBuffer(MlSkin * skin)
{
	const unsigned num = skin->numCreated();
	if(num < 1) return;
	
	unsigned loc = taken();
	
	unsigned i, nvpf;
	Vector3F v;
	for(i = 0; i < num; i++) {
		MlCalamus * c = skin->getCreated(i);
		
		m_featherTess->setFeather(c->feather());
		
		setIndex(loc);
		c->setBufferStart(loc);
		
		nvpf = m_featherTess->numIndices();

		expandBy(nvpf);
		
		loc += nvpf;
	}
}

void MlDrawer::readBuffer(MlSkin * skin)
{
	this->skin = skin;
	const unsigned nc = skin->numFeathers();
	if(nc < 1) return;
	
	std::stringstream sst;
	sst.str("");
	sst<<m_currentFrame;
	
	if(!isCached("/p", sst.str())) return;
	
	useDocument();
	openEntry("/p");
	openSlice("/p", sst.str());
	
	readFromCache(sst.str());
		
	closeSlice("/p", sst.str());
	closeEntry("/p");
}

void MlDrawer::rebuildBuffer(MlSkin * skin, bool forced)
{
	this->skin = skin;
	const unsigned nc = skin->numFeathers();
	if(nc < 1) return;
	
	std::stringstream sst;
	sst.str("");
	sst<<m_currentFrame;
	
	useDocument();
	openEntry("/p");
	openSlice("/p", sst.str());
	
	if(isCached("/p", sst.str()) && !forced)
		readFromCache(sst.str());
	else
		writeToCache(sst.str());
		
	closeSlice("/p", sst.str());
	closeEntry("/p");
}

void MlDrawer::computeBufferIndirection(MlSkin * skin)
{
	const unsigned nc = skin->numFeathers();
	unsigned i, nvpf;
	unsigned loc = 0;
	for(i = 0; i < nc; i++) {
		MlCalamus * c = skin->getCalamus(i);
		
		m_featherTess->setFeather(c->feather());
		
		setIndex(loc);
		c->setBufferStart(loc);
		
		nvpf = m_featherTess->numIndices();

		expandBy(nvpf);
		
		loc += nvpf;
	}

	std::cout<<"buffer n vertices: "<<loc<<"\n";
}

void MlDrawer::computeFeather(MlSkin * skin, MlCalamus * c)
{
	skin->touchBy(c);

	Vector3F p;
	skin->getPointOnBody(c, p);
	
	Matrix33F space;
	skin->calamusSpace(c, space);
	c->bendFeather(p, space);
	c->curlFeather();
	c->computeFeatherWorldP(p, space);
}

void MlDrawer::computeFeather(MlSkin * skin, MlCalamus * c, const Vector3F & p, const Matrix33F & space)
{
	skin->touchBy(c);
	c->bendFeather(p, space);
	c->curlFeather();
	c->computeFeatherWorldP(p, space);
}

void MlDrawer::tessellate(MlFeather * f)
{
	m_featherTess->setFeather(f);
	m_featherTess->evaluate(f);
}

void MlDrawer::writeToCache(const std::string & sliceName)
{
	skin->computeClusterSamples();
	
	unsigned faceIdx = skin->bodyMesh()->getNumFaces();
	unsigned perFaceIdx = 0;
	const unsigned nc = skin->numFeathers();
	const unsigned blockL = 2048;
	Vector3F * wpb = new Vector3F[blockL];
	unsigned i, j, iblock = 0, ifull = 0;
	BoundingBox box;
	Matrix33F space;
	Vector3F p;
	unsigned ncalc = 0;
	unsigned nsamp = 0;
	unsigned nreuse = 0;
	for(i = 0; i < nc; i++) {
		MlCalamus * c = skin->getCalamus(i);
		skin->calamusSpace(c, space);
		skin->getPointOnBody(c, p);
		
		if(c->faceIdx() != faceIdx) {
			faceIdx = c->faceIdx();
			perFaceIdx = 0;
			nsamp += skin->clusterK(faceIdx);
		}
		
		if(skin->useClusterSamples(faceIdx, perFaceIdx, c, space)) {
			c->bendFeather();
			c->curlFeather();
			c->computeFeatherWorldP(p, space);
			nreuse++;
		}
		else {
			computeFeather(skin, c, p, space);
			ncalc++;
		}
		
		MlFeather * f = c->feather();
		f->getBoundingBox(box);
		for(j = 0; j < f->numWorldP(); j++) {
			wpb[iblock] = f->worldP()[j];
			iblock++;
			ifull++;
			if(iblock == blockL) {
				writeSliceVector3("/p", sliceName, ifull - iblock, iblock, wpb);
				iblock = 0;
			}
		}
		
		updateBuffer(c);
		
		perFaceIdx++;
	}
	
	if(iblock > 0)
		writeSliceVector3("/p", sliceName, ifull - iblock, iblock, wpb);
		
	saveEntrySize("/p", ifull);
	setCached("/p", sliceName, ifull);
	setBounding(sliceName, box);
	setTranslation(sliceName, m_currentOrigin);
	delete[] wpb;
	flush();
	
	std::cout<<" full "<<nc<<" calc "<< (float)(nsamp + ncalc) / (float)nc;
}

void MlDrawer::readFromCache(const std::string & sliceName)
{
	const unsigned nc = skin->numFeathers();
	const unsigned blockL = 2048;
	Vector3F * wpb = new Vector3F[blockL];
	unsigned i, j, iblock = 0, ifull = 0;
	readSliceVector3("/p", sliceName, 0, blockL, wpb);
	
	for(i = 0; i < nc; i++) {
		
		MlCalamus * c = skin->getCalamus(i);
		
		MlFeather * f = c->feather();
		
		Vector3F * dst = f->worldP();
		for(j = 0; j < f->numWorldP(); j++) {
			dst[j] = wpb[iblock];
			iblock++;
			ifull++;
			if(iblock == blockL) {
				readSliceVector3("/p", sliceName, ifull, blockL, wpb);
				iblock = 0;
			}
		}
		
		updateBuffer(c);
	}
	delete[] wpb;
}

void MlDrawer::setCurrentFrame(int x)
{
	m_currentFrame = x;
}

void MlDrawer::setCurrentOrigin(const Vector3F & at)
{
    m_currentOrigin = at;
}

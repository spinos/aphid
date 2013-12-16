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
#include <boost/timer.hpp>
MlDrawer::MlDrawer()
{
	std::cout<<"Feather buffer ";
	m_featherTess = new MlTessellate;
	m_currentFrame = -9999;
	m_skin = 0;
}

MlDrawer::~MlDrawer() 
{
	delete m_featherTess;
}

void MlDrawer::setSkin(MlSkin * skin)
{
	m_skin = skin;
}

void MlDrawer::draw() const
{
	if(!m_skin) return;
	if(m_skin->numFeathers() > 0) drawBuffer();
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

void MlDrawer::hideActive()
{
	const unsigned num = m_skin->numActive();
	if(num < 1) return;
	
	unsigned i;
	for(i = 0; i < num; i++) {
		MlCalamus * c = m_skin->getActive(i);
		hideAFeather(c);
	}
}

void MlDrawer::updateActive()
{
	const unsigned num = m_skin->numActive();
	if(num < 1) return;
	
	unsigned i;
	for(i = 0; i < num; i++) {
		MlCalamus * c = m_skin->getActive(i);
		computeFeather(c);
		updateBuffer(c);
	}
	disable();
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

void MlDrawer::addToBuffer()
{
	const unsigned num = m_skin->numCreated();
	if(num < 1) return;
	
	unsigned loc = taken();
	
	unsigned i, nvpf;
	Vector3F v;
	for(i = 0; i < num; i++) {
		MlCalamus * c = m_skin->getCreated(i);
		
		m_featherTess->setFeather(c->feather());
		
		setIndex(loc);
		c->setBufferStart(loc);
		
		nvpf = m_featherTess->numIndices();

		expandBy(nvpf);
		
		loc += nvpf;
	}
}

void MlDrawer::readBuffer()
{
	const unsigned nc = m_skin->numFeathers();
	if(nc < 1) return;
	
	std::stringstream sst;
	sst.str("");
	sst<<m_currentFrame;
	
	if(!isCached("/ang", sst.str())) return;
	
	useDocument();
	openEntry("/ang");
	openSliceFloat("/ang", sst.str());
	openEntry("/p");
	openSliceVector3("/p", sst.str());
	openEntry("/tang");
	openSliceMatrix33("/tang", sst.str());
	
	readFromCache(sst.str());
		
	closeSlice("/ang", sst.str());
	closeEntry("/ang");
	closeSlice("/p", sst.str());
	closeEntry("/p");
	closeSlice("/tang", sst.str());
	closeEntry("/tang");
}

void MlDrawer::rebuildBuffer(MlSkin * skin)
{
	m_skin = skin;
	const unsigned nc = skin->numFeathers();
	if(nc < 1) return;
	
	if(!isEnabled()) {
		rebuildIgnoreCache();
		return;
	}
	
	std::stringstream sst;
	sst.str("");
	sst<<m_currentFrame;
	
	useDocument();
	openEntry("/ang");
	openSliceFloat("/ang", sst.str());
	openEntry("/p");
	openSliceVector3("/p", sst.str());
	openEntry("/tang");
	openSliceMatrix33("/tang", sst.str());
	
	if(isCached("/ang", sst.str()))
		readFromCache(sst.str());
	else
		writeToCache(sst.str());
		
	closeSlice("/ang", sst.str());
	closeEntry("/ang");
	closeSlice("/p", sst.str());
	closeEntry("/p");
	closeSlice("/tang", sst.str());
	closeEntry("/tang");
}

void MlDrawer::computeBufferIndirection()
{
	if(!m_skin) return;
	const unsigned nc = m_skin->numFeathers();
	unsigned i, nvpf;
	unsigned loc = 0;
	for(i = 0; i < nc; i++) {
		MlCalamus * c = m_skin->getCalamus(i);
		
		m_featherTess->setFeather(c->feather());
		
		setIndex(loc);
		c->setBufferStart(loc);
		
		nvpf = m_featherTess->numIndices();

		expandBy(nvpf);
		
		loc += nvpf;
	}

	std::cout<<"buffer n vertices: "<<loc<<"\n";
}

void MlDrawer::computeFeather(MlCalamus * c)
{
	Vector3F p;
	m_skin->getPointOnBody(c, p);
	
	Matrix33F space;
	m_skin->calamusSpace(c, space);
	m_skin->touchBy(c, p, space);
	c->bendFeather(p, space);
	c->curlFeather();
	c->computeFeatherWorldP(p, space);
}

void MlDrawer::computeFeather(MlCalamus * c, const Vector3F & p, const Matrix33F & space)
{
	m_skin->touchBy(c, p, space);
	c->bendFeather(p, space);
	c->curlFeather();
	c->computeFeatherWorldP(p, space);
}

void MlDrawer::tessellate(MlFeather * f)
{
	m_featherTess->setFeather(f);
	m_featherTess->evaluate(f);
}

void MlDrawer::rebuildIgnoreCache()
{
    boost::timer bTimer;
	bTimer.restart();
	m_skin->computeFaceClustering();
	m_skin->computeClusterSamples();
	
	unsigned faceIdx = m_skin->bodyMesh()->getNumFaces();
	unsigned perFaceIdx = 0;
	const unsigned nc = m_skin->numFeathers();
	unsigned i;
	Matrix33F space;
	Vector3F p;
	unsigned ncalc = 0;
	unsigned nsamp = 0;
	unsigned nreuse = 0;
	for(i = 0; i < nc; i++) {
		MlCalamus * c = m_skin->getCalamus(i);
		m_skin->calamusSpace(c, space);
		m_skin->getPointOnBody(c, p);
		
		if(c->faceIdx() != faceIdx) {
			faceIdx = c->faceIdx();
			perFaceIdx = 0;
			nsamp += m_skin->clusterK(faceIdx);
		}
		
		if(m_skin->useClusterSamples(faceIdx, perFaceIdx, c, i)) {
			c->bendFeather();
			c->curlFeather();
			c->computeFeatherWorldP(p, space);
			nreuse++;
		}
		else {
			computeFeather(c, p, space);
			ncalc++;
		}
		
		updateBuffer(c);
		
		perFaceIdx++;
	}
	
	std::cout<<" sample "<< (float)(nsamp + ncalc) / (float)nc * 100 <<"% in "<<bTimer.elapsed()<<" seconds\n";
}

void MlDrawer::writeToCache(const std::string & sliceName)
{
    boost::timer bTimer;
	bTimer.restart();
	const unsigned nc = m_skin->numFeathers();
	const unsigned blockL = 4096;
	Vector3F * wpb = new Vector3F[blockL];
	unsigned i, iblock = 0, ifull = 0, ipblock = 0, ipfull = 0;
	short j;
	float * apb = new float[blockL];
	
	Matrix33F * tangs = new Matrix33F[blockL];
	
	BoundingBox box;
	Matrix33F space;
	Vector3F p;

	for(i = 0; i < nc; i++) {
		MlCalamus * c = m_skin->getCalamus(i);
		m_skin->calamusSpace(c, space);
		m_skin->getPointOnBody(c, p);
		
		wpb[ipblock] = p;
		tangs[ipblock] = space;
		ipblock++;
		ipfull++;
		if(ipblock == blockL) {
			writeSliceVector3("/p", sliceName, ipfull - ipblock, ipblock, wpb);
			writeSliceMatrix33("/tang", sliceName, ipfull - ipblock, ipblock, tangs);
			ipblock = 0;
		}
		
		computeFeather(c, p, space);

		MlFeather * f = c->feather();
		f->getBoundingBox(box);
		
		float * src = f->angles();
		for(j = 0; j < f->numSegment(); j++) {
			apb[iblock] = src[j];
			iblock++;
			ifull++;
			if(iblock == blockL) {
				writeSliceFloat("/ang", sliceName, ifull - iblock, iblock, apb);
				iblock = 0;
			}
		}

		updateBuffer(c);
	}
	
	if(ipblock > 0) {
		writeSliceVector3("/p", sliceName, ipfull - ipblock, ipblock, wpb);
		writeSliceMatrix33("/tang", sliceName, ipfull - ipblock, ipblock, tangs);
	}
	
	if(iblock > 0)
		writeSliceFloat("/ang", sliceName, ifull - iblock, iblock, apb);
		
	saveEntrySize("/p", ipfull);
	setCached("/p", sliceName, ipfull);
	saveEntrySize("/tang", ipfull);
	setCached("/tang", sliceName, ipfull);
	saveEntrySize("/ang", ifull);
	setCached("/ang", sliceName, ifull);
	setBounding(sliceName, box);
	setTranslation(sliceName, m_currentOrigin);
	delete[] wpb;
	delete[] apb;
	delete[] tangs;
	flush();
	
	std::cout<<" write "<< sliceName <<" in "<<bTimer.elapsed()<<" seconds\n";
}

void MlDrawer::readFromCache(const std::string & sliceName)
{
	const unsigned nc = m_skin->numFeathers();
	const unsigned blockL = 4096;
	unsigned i, iblock = 0, ifull = 0;
	unsigned ipblock = 0, ipfull = 0;
	short j;
	float * apb = new float[blockL];
	readSliceFloat("/ang", sliceName, 0, blockL, apb);
	Vector3F * wpb = new Vector3F[blockL];
	readSliceVector3("/p", sliceName, 0, blockL, wpb);
	Matrix33F * tangs = new Matrix33F[blockL];
	readSliceMatrix33("/tang", sliceName, 0, blockL, tangs);

	Matrix33F space;
	Vector3F p;
	for(i = 0; i < nc; i++) {
		MlCalamus * c = m_skin->getCalamus(i);
		space = tangs[ipblock];
		p = wpb[ipblock];
		ipblock++;
		ipfull++;
		if(ipblock == blockL) {
			readSliceVector3("/p", sliceName, ipfull, blockL, wpb);
			readSliceMatrix33("/tang", sliceName, ipfull, blockL, tangs);
			ipblock = 0;
		}
		
		MlFeather * f = c->feather();
		
		float *dst = f->angles();
		
		for(j = 0; j < f->numSegment(); j++) {
			dst[j] = apb[iblock];
			iblock++;
			ifull++;
			if(iblock == blockL) {
				readSliceFloat("/ang", sliceName, ifull, blockL, apb);
				iblock = 0;
			}
		}
		
		c->bendFeather();
		c->curlFeather();
		c->computeFeatherWorldP(p, space);
		
		updateBuffer(c);
	}
	delete[] apb;
	delete[] wpb;
	delete[] tangs;
}

void MlDrawer::setCurrentFrame(int x)
{
	m_currentFrame = x;
}

void MlDrawer::setCurrentOrigin(const Vector3F & at)
{
    m_currentOrigin = at;
}

int MlDrawer::currentFrame() const
{
	return m_currentFrame;
}

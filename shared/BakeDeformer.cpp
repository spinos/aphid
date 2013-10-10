/*
 *  BakeDeformer.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BakeDeformer.h"
#include <AllHdf.h>
#include <HBase.h>
#include <sstream>
BakeDeformer::BakeDeformer()
{}

BakeDeformer::~BakeDeformer() 
{
    clearFrames();
}

bool BakeDeformer::open(const std::string & filename)
{
	if(!HObject::FileIO.open(filename.c_str(), HDocument::oReadOnly)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	bool found = false;
	HBase g("/");
	int nf = g.numChildren();
	for(int i = 0; i < nf; ++i) {
		if(!g.isChildGroup(i)) continue;
		HBase c(g.childPath(i));
		found = findMatchedMesh(c);
		c.close();
		if(found) break;
	}
	g.close();
	
	if(found) {
		processFrameRange();
		processFrameCenters();
	}
	
	HObject::FileIO.close();
	
	if(!found) {
		std::cout<<"ERROR: cannot find matched bake cache in file "<<filename<<"\n";
		return 0;
	}
	
	return BaseFile::open(filename);
}

bool BakeDeformer::findMatchedMesh(HBase & grp)
{
	bool found = false;
	if(grp.hasNamedChild(".geom")) {
		
		//std::cout<<grp.fObjectPath<<" is mesh\n";
		
		HBase g(grp.childPath(".geom"));
		found = isGeomMatched(g);
		g.close();
		return found;
	}
	
	int nf = grp.numChildren();
	for(int i = 0; i < nf; ++i) {
		//std::cout<<grp.getChildName(i);
		if(grp.isChildGroup(i)) {
			HBase c(grp.childPath(i));
			found = findMatchedMesh(c);
			c.close();
			if(found) return true;
		}
	}
	return false;
}

bool BakeDeformer::isGeomMatched(HBase & grp)
{
	if(!grp.hasNamedChild(".bake")) return false;
	
	//std::cout<<grp.fObjectPath<<" has bake\n";
	
	bool found = false;
	
	HBase b(grp.childPath(".bake"));
	
	found = isBakeMatched(b);
	
	b.close();
	
	return found;
}

bool BakeDeformer::isBakeMatched(HBase & grp)
{
	if(grp.numChildren() < 1) return false;
	HDataset s(grp.getChildName(0));
	s.open(grp.fObjectId);
	int dims[3];
	int ndim;
	s.getSpaceDimension(dims, &ndim);
	
	s.close();
	
	m_bakePath = grp.fObjectPath;
	return ((unsigned)dims[0] == numVertices() * 3);
}

void BakeDeformer::processFrameRange()
{
	m_minFrame = 10e6;
	m_maxFrame = -10e6;
	int iframe;
	HBase b(m_bakePath);
	int n = b.numChildren();
	for(int i = 0; i < n; i++) {
		std::string sframe(b.getChildName(i));
		std::istringstream(sframe) >> iframe;
		if(iframe > m_maxFrame) m_maxFrame = iframe;
		if(iframe < m_minFrame) m_minFrame = iframe;
	}
	b.close();
}

void BakeDeformer::processFrameCenters()
{
    clearFrames();

    int iframe;
	HBase b(m_bakePath);
	int n = b.numChildren();
	for(int i = 0; i < n; i++) {
		std::string sframe(b.getChildName(i));
		std::istringstream(sframe) >> iframe;
		
		b.readVector3Data(b.getChildName(i).c_str(), numVertices(), deformedP());
		
		BoundingBox b = calculateBBox();
		
		m_frameCenters[iframe] = b.center();
	}
	b.close();
	
	m_currentFrame = m_minFrame;
}

void BakeDeformer::clearFrames()
{
    if(m_frameCenters.size() > 0)
        m_frameCenters.clear();
}

char BakeDeformer::solve()
{
    if(!isValid()) return 0;
	if(!isEnabled()) return 0;
    
    if(!HObject::FileIO.open(fileName().c_str(), HDocument::oReadOnly)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	//std::cout<<"read bake at frame "<<m_currentFrame<<"\n";
	HBase b(m_bakePath);
	std::stringstream sst;
	sst.str("");
	sst<<m_currentFrame;
	char status = b.readVector3Data(sst.str().c_str(), numVertices(), deformedP());
	
	HObject::FileIO.close();
	
	if(status) {
	    Vector3F c = m_frameCenters[m_currentFrame];
	    for(unsigned i = 0; i < numVertices(); i++)
	        deformedP()[i] -= c;
	}
    return status;
}

void BakeDeformer::setCurrentFrame(int x)
{
    if(x < m_minFrame) m_currentFrame = m_minFrame;
    else if(x > m_maxFrame) m_currentFrame = m_maxFrame;
    else m_currentFrame = x;
}

int BakeDeformer::minFrame() const
{
	return m_minFrame;
}

int BakeDeformer::maxFrame() const
{
	return m_maxFrame;
}

void BakeDeformer::verbose() const
{
	std::cout<<"Bake Cache Deformer: "<<fileName()<<"\n num vertices "<<numVertices();
	std::cout<<" frame range "<<m_minFrame<<" to "<<m_maxFrame<<"\n";
	std::cout<<" bake path "<<m_bakePath<<"\n";
}

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
BakeDeformer::BakeDeformer() {}
BakeDeformer::~BakeDeformer() {}

char BakeDeformer::load(const char * filename)
{
	if(!HObject::FileIO.open(filename, HDocument::oReadOnly)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	std::cout<<"read bake from "<<filename<<"\n";
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
		std::cout<<"Found matched bake cache\n";
		std::cout<<" frame range: "<<m_minFrame<<" to "<<m_maxFrame<<"\n";
		std::cout<<" bake path: "<<m_bakePath<<"\n";
	}
	
	HObject::FileIO.close();
	
	if(!found) {
		std::cout<<"ERROR: cannot find matched bake cache\n";
		return 0;
	}
	
	return BaseFile::load(filename);
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

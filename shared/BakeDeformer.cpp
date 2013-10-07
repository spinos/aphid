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

void scan(HBase & g)
{
	std::cout<<"attrs of group "<<g.fObjectPath<<"\n";
	int na = g.numAttrs();
	std::cout<<" n attrs "<<na<<"\n";
	for(int i = 0; i < na; ++i) {
		std::cout<<" "<<g.getAttrName(i)<<"\n";
	}
	
	int nf = g.numChildren();
	for(int i = 0; i < nf; ++i) {
		std::stringstream sst;
		sst.str("");
		sst<<g.fObjectPath<<"/"<<g.getChildName(i);
		std::cout<<g.getChildName(i);
		if(g.isChildGroup(i)) {
			std::cout<<" is group\n";
			HBase c(sst.str());
			scan(c);
			c.close();
		}
		else if(g.isChildData(i)) {
			std::cout<<" is data\n";
			HDataset s(g.getChildName(i));
			s.open(g.fObjectId);
			int dims[3];
			int ndim;
			s.getSpaceDimension(dims, &ndim);
			std::cout<<" dims "<<dims[0]<<"\n";
			s.close();
		}
	}
}

char BakeDeformer::load(const char * filename)
{
	if(!HObject::FileIO.open(filename, HDocument::oReadOnly)) {
		setLatestError(BaseFile::FileNotReadable);
		return false;
	}
	
	std::cout<<"read bake from "<<filename<<"\n";
	
	HBase g("/");
	int nf = g.numChildren();
	for(int i = 0; i < nf; ++i) {
		if(g.isChildGroup(i)) {
			std::stringstream sst;
			sst.str("");
			sst<<"/"<<g.getChildName(i);
			HBase c(sst.str());
			scan(c);
			c.close();
		}
	}
	g.close();
	
	HObject::FileIO.close();
	return BaseFile::load(filename);
}
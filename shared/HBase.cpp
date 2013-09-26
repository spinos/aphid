/*
 *  HBase.cpp
 *  masq
 *
 *  Created by jian zhang on 5/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HBase.h"

#include <sstream>
#include <iostream>
HBase::HBase(const std::string & path) : HGroup(path) 
{
	if(!HObject::FileIO.checkExist(fObjectPath))
		create();
	else 
		open();
}

HBase::~HBase() {}

void HBase::addIntAttr(const char * attrName, int *value)
{
	HIntAttribute nvAttr(attrName);
	nvAttr.create(1, fObjectId);
	if(!nvAttr.write(value)) std::cout<<attrName<<" write failed";
	nvAttr.close();
}

void HBase::writeIntAttr(const char * attrName, int *value)
{
	HIntAttribute nvAttr(attrName);
	nvAttr.open(fObjectId);
	if(!nvAttr.write(value)) std::cout<<attrName<<" write failed";
	nvAttr.close();
}

void HBase::addIntData(const char * dataName, unsigned count, int *value)
{	
	IndicesHDataset cset(dataName);
	cset.setNumIndices(count);
	cset.create();
	cset.close();
	cset.open();
	if(!cset.write((int *)value)) std::cout<<dataName<<" write failed";
	cset.close();
}

void HBase::addVector3Data(const char * dataName, unsigned count, Vector3F *value)
{	
	VerticesHDataset pset(dataName);
	pset.setNumVertices(count);
	pset.create();
	pset.close();
	pset.open();
	if(!pset.write((float *)value)) std::cout<<dataName<<" write failed";
	pset.close();
}

char HBase::readIntAttr(const char * dataName, int *value)
{	
	HIntAttribute nvAttr(dataName);
	if(!nvAttr.open(fObjectId)) {
		std::cout<<dataName<<" open failed";
		return 0;
	}
	
	if(!nvAttr.read(value)) {
		std::cout<<dataName<<" read failed";
		return 0;
	}

	nvAttr.close();
	
	return 1;
}

char HBase::readIntData(const char * dataName, unsigned count, unsigned *dst)
{	
	IndicesHDataset cset(dataName);
	cset.setNumIndices(count);
	
	if(!cset.open()) {
		std::cout<<dataName<<" open failed";
		return 0;
	}
	
	if(cset.dimensionMatched()) {
		cset.read((int *)dst);
	}
	else {
		std::cout<<dataName<<" dim check failed";
		return 0;
	}
		
	cset.close();
	return 1;
}

char HBase::readVector3Data(const char * dataName, unsigned count, Vector3F *dst)
{
	VerticesHDataset pset(dataName);
	pset.setNumVertices(count);
	
	if(!pset.open()) {
		std::cout<<dataName<<" open failed";
		return 0;
	}
	
	if(pset.dimensionMatched()) {
		pset.read((float *)dst);
	}
	else {
		std::cout<<dataName<<" dim check failed";
		return 0;
	}
	
	pset.close();
	return 1;
}

char HBase::hasNamedAttr(const char * attrName)
{
	hsize_t nattr = H5Aget_num_attrs(fObjectId);
	std::cout<<"\n "<<fObjectPath<<" has "<<nattr<<" attrs\n";
	hsize_t i;
	for(i = 0; i < nattr; i++) {
		hid_t aid = H5Aopen_idx(fObjectId, (unsigned int)i );
		std::cout<<getAttrName(aid)<<"\n";
		if(getAttrName(aid) == attrName) {
			std::cout<<"found "<<attrName;
			return 1;
		}
	}
	return 0;
}

std::string HBase::getAttrName(hid_t attrId)
{
	char memb_name[1024];
	H5Aget_name(attrId, (size_t)1024, 
		memb_name );
	std::stringstream sst;
	sst.str("");
	sst<<memb_name;
	return sst.str();
}

char HBase::hasNamedChild(const char * childName)
{
	hsize_t nobj;
	H5Gget_num_objs(fObjectId, &nobj);
	std::cout<<"\n"<<fObjectPath<<" has "<<nobj<<"objs\n";
	hsize_t i;
	for(i = 0; i < nobj; i++) {
		std::cout<<getChildName(i)<<"\n";
		if(getChildName(i) == childName) {
			std::cout<<"found "<<childName;
			return 1;
		}
	}
	return 0;
}

std::string HBase::getChildName(hsize_t i)
{
	char memb_name[1024];
	H5Gget_objname_by_idx(fObjectId, (hsize_t)i, 
		memb_name, (size_t)1024 );
	std::stringstream sst;
	sst.str("");
	sst<<memb_name;
	return sst.str();
}

char HBase::save()
{
	return 1;
}

char HBase::load()
{
	return 1;
}

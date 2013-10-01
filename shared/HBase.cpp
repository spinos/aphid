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

void HBase::addIntAttr(const char * attrName)
{
	HIntAttribute nvAttr(attrName);
	nvAttr.create(1, fObjectId);
	nvAttr.close();
}

void HBase::addIntData(const char * dataName, unsigned count)
{	
	IndicesHDataset cset(dataName);
	cset.setNumIndices(count);
	cset.create(fObjectId);
	cset.close();
}

void HBase::addFloatData(const char * dataName, unsigned count)
{
	FloatsHDataset cset(dataName);
	cset.setNumFloats(count);
	cset.create(fObjectId);
	cset.close();
}

void HBase::addVector3Data(const char * dataName, unsigned count)
{	
	VerticesHDataset pset(dataName);
	pset.setNumVertices(count);
	pset.create(fObjectId);
	pset.close();
}

void HBase::addCharData(const char * dataName, unsigned count)
{
	HCharData cset(dataName);
	cset.setNumChars(count);
	cset.create(fObjectId);
	cset.close();
}

void HBase::writeIntAttr(const char * attrName, int *value)
{
	HIntAttribute nvAttr(attrName);
	nvAttr.open(fObjectId);
	if(!nvAttr.write(value)) std::cout<<attrName<<" write failed";
	nvAttr.close();
}

void HBase::writeIntData(const char * dataName, unsigned count, int *value, HDataset::SelectPart * part)
{	
	IndicesHDataset cset(dataName);
	cset.setNumIndices(count);
	cset.open(fObjectId);

	if(!cset.write((char *)value, part)) std::cout<<dataName<<" write failed";
	cset.close();
}

void HBase::writeFloatData(const char * dataName, unsigned count, float *value, HDataset::SelectPart * part)
{	
	FloatsHDataset cset(dataName);
	cset.setNumFloats(count);
	cset.open(fObjectId);

	if(!cset.write((char *)value, part)) std::cout<<dataName<<" write failed";
	cset.close();
}

void HBase::writeVector3Data(const char * dataName, unsigned count, Vector3F *value, HDataset::SelectPart * part)
{
    VerticesHDataset pset(dataName);
	pset.setNumVertices(count);	
	pset.open(fObjectId);

	if(!pset.write((char *)value, part)) std::cout<<dataName<<" write failed";
	pset.close();
}

void HBase::writeCharData(const char * dataName, unsigned count, char *value, HDataset::SelectPart * part)
{
	HCharData pset(dataName);
	pset.setNumChars(count);	
	pset.open(fObjectId);

	if(!pset.write((char *)value, part)) std::cout<<dataName<<" write failed";
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

char HBase::readIntData(const char * dataName, unsigned count, unsigned *dst, HDataset::SelectPart * part)
{	
	IndicesHDataset cset(dataName);
	cset.setNumIndices(count);
	
	if(!cset.open(fObjectId)) {
		std::cout<<dataName<<" open failed";
		return 0;
	}
	
	cset.read((char *)dst, part);
		
	cset.close();
	return 1;
}

char HBase::readFloatData(const char * dataName, unsigned count, float *dst, HDataset::SelectPart * part)
{	
	FloatsHDataset cset(dataName);
	cset.setNumFloats(count);
	
	if(!cset.open(fObjectId)) {
		std::cout<<dataName<<" open failed";
		return 0;
	}
	
	cset.read((char *)dst, part);
		
	cset.close();
	return 1;
}

char HBase::readVector3Data(const char * dataName, unsigned count, Vector3F *dst, HDataset::SelectPart * part)
{
	VerticesHDataset pset(dataName);
	pset.setNumVertices(count);
	
	if(!pset.open(fObjectId)) {
		std::cout<<dataName<<" open failed";
		return 0;
	}
	
	pset.read((char *)dst, part);
	pset.close();
	return 1;
}

char HBase::readCharData(const char * dataName, unsigned count, char *dst, HDataset::SelectPart * part)
{
	HCharData pset(dataName);
	pset.setNumChars(count);
	if(!pset.open(fObjectId)) {
		std::cout<<dataName<<" open failed";
		return 0;
	}
	pset.read((char *)dst, part);
	pset.close();
	return 1;
}

char HBase::hasNamedAttr(const char * attrName)
{
	hsize_t nattr = H5Aget_num_attrs(fObjectId);
	//std::cout<<"\n "<<fObjectPath<<" has "<<nattr<<" attrs\n";
	hsize_t i;
	for(i = 0; i < nattr; i++) {
		hid_t aid = H5Aopen_idx(fObjectId, (unsigned int)i );
		//std::cout<<getAttrName(aid)<<"\n";
		if(getAttrName(aid) == attrName) {
			//std::cout<<"found "<<attrName;
			H5Aclose(aid);
			return 1;
		}
		H5Aclose(aid);
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
	hsize_t nobj = 0;
	H5Gget_num_objs(fObjectId, &nobj);
	//std::cout<<"\n"<<fObjectPath<<" has "<<nobj<<"objs\n";
	hsize_t i;
	for(i = 0; i < nobj; i++) {
		//std::cout<<getChildName(i)<<"\n";
		if(getChildName(i) == childName) {
			//std::cout<<"found "<<childName;
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

char HBase::hasNamedData(const char * dataName)
{
	hsize_t nobj;
	int otype;
	H5Gget_num_objs(fObjectId, &nobj);
	//std::cout<<"\n"<<fObjectPath<<" has "<<nobj<<"objs\n";
	hsize_t i;
	for(i = 0; i < nobj; i++) {
		//std::cout<<getChildName(i)<<"\n";
		
		otype =  H5Gget_objtype_by_idx(fObjectId, (size_t)i );
		
		if(otype == H5G_DATASET) {
            if(getChildName(i) == dataName) {
                //std::cout<<"found "<<dataName<<"\n";;
                return 1;
            }
        }
	}
	return 0;
}

char HBase::discardNamedAttr(const char * path)
{
    if(H5Adelete(fObjectId, path) < 0) return 0;
    return 1;
}

int HBase::numChildren()
{
	hsize_t nobj = 0;
	H5Gget_num_objs(fObjectId, &nobj);
	return (int)nobj;
}

int HBase::numAttrs()
{
	hsize_t nattr = H5Aget_num_attrs(fObjectId);
	return (int)nattr;
}

char HBase::save()
{
	return 1;
}

char HBase::load()
{
	return 1;
}

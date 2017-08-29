#ifndef HBASE_H
#define HBASE_H

/*
 *  HBase.h
 *  masq
 *
 *  Created by jian zhang on 5/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AllHdf.h"
#include <string>
#include <iostream>
#include <vector>

namespace aphid {

class Vector3F;
class Matrix33F;
class Matrix44F;

class HBase : public HGroup {
public:
	HBase(const std::string & path);
	virtual ~HBase();
	
	void addIntAttr(const char * attrName, int dim = 1);
	void addFloatAttr(const char * attrName, int dim = 1);
	void addStringAttr(const char * attrName, int dim);
	void addIntData(const char * dataName, unsigned count);
	void addFloatData(const char * dataName, unsigned count);
	void addVector3Data(const char * dataName, unsigned count);
	void addCharData(const char * dataName, unsigned count);
	void addVLStringAttr(const char * attrName);
	
	void writeIntAttr(const char * attrName, int *value);
	void writeFloatAttr(const char * attrName, float *value);
	void writeStringAttr(const char * attrName, const std::string & value);
	void writeIntData(const char * dataName, unsigned count, int *value, HDataset::SelectPart * part = 0);
	void writeFloatData(const char * dataName, unsigned count, float *value, HDataset::SelectPart * part = 0);
	void writeVector3Data(const char * dataName, unsigned count, Vector3F *value, HDataset::SelectPart * part = 0);
	void writeMatrix33Data(const char * dataName, unsigned count, Matrix33F *value, HDataset::SelectPart * part = 0);
	void writeMatrix44Data(const char * dataName, unsigned count, Matrix44F *value, HDataset::SelectPart * part = 0);
	void writeCharData(const char * dataName, unsigned count, char *value, HDataset::SelectPart * part = 0);
	void writeVLStringAttr(const char * attrName, const std::string & value);
	
	char readIntAttr(const char * attrName, int *value);
	char readFloatAttr(const char * attrName, float *value);
	char readStringAttr(const char * attrName, std::string & value);
	char readIntData(const char * dataname, unsigned count, int *dst, HDataset::SelectPart * part = 0);
	char readFloatData(const char * dataname, unsigned count, float *dst, HDataset::SelectPart * part = 0);
	char readVector3Data(const char * dataname, unsigned count, Vector3F *dst, HDataset::SelectPart * part = 0);
	char readMatrix33Data(const char * dataname, unsigned count, Matrix33F *dst, HDataset::SelectPart * part = 0);
	char readCharData(const char * dataName, unsigned count, char *dst, HDataset::SelectPart * part = 0);
	char readVLStringAttr(const char * attrName, std::string & value);
	
	char hasNamedAttr(const char * attrName);
	std::string getAttrName(hid_t attrId);
	
	char hasNamedChild(const char * childName);
	std::string getChildName(hsize_t i);
	bool isChildGroup(hsize_t i);
	bool isChildData(hsize_t i);
	
	char hasNamedData(const char * dataName);
	char discardNamedAttr(const char * path);
	
	bool hasNamedAttrIntVal(const std::string & attrName,
							int attrVal);
	
	int numChildren();
	int numAttrs();
	int numDatas();
	
	std::string childPath(const std::string & name) const;
	std::string childPath(int i);
	
	virtual char save();
	virtual char load();
    
    virtual char verifyType();

	template<typename T>
	void lsTypedChild(std::vector<std::string> & names) {
		int nc = numChildren();
		int i = 0;
		for(;i<nc;i++) {
			if(isChildGroup(i)) {
				T gc(childPath(i));
				if(gc.verifyType()) names.push_back(childPath(i));
				gc.close();
			}
		}
	}
	
	template<typename T>
	void lsTypedChildWithIntAttrVal(std::vector<std::string> & names,
									const std::string & attrName,
									int attrVal ) {
		int nc = numChildren();
		int i = 0;
		for(;i<nc;i++) {
			if(isChildGroup(i)) {
				T gc(childPath(i));
				if(gc.verifyType()) {
					if(gc.hasNamedAttrIntVal(attrName, attrVal ) )
						names.push_back(childPath(i));
				}
				gc.close();
			}
		}
	}
	
	template<typename T>
	bool hasTypedChildWithIntAttrVal(const std::string & name,
									const std::string & attrName,
									int attrVal ) 
	{
		bool stat = false;
		if(hasNamedChild(name.c_str() ) ) {
			T gc(childPath(name ) );
			if(gc.verifyType() ) {
				if(gc.hasNamedAttrIntVal(attrName, attrVal ) )
					stat = true;
			}
			gc.close();
		}
		return stat;
	}
    
    void lsData(std::vector<std::string> & names) {
        int nc = numChildren();
		int i = 0;
		for(;i<nc;i++) {
			if(isChildData(i)) {
				names.push_back(childPath(i));
			}
		}
    }
	
	template<typename T>
	T * createDataStorage(const std::string & name, bool toClear, bool & stat)
	{
		T * d = new T(name);
		if(hasNamedData(name.c_str() ) ) {
			stat = d->openStorage(fObjectId, false);
			if(stat) stat = d->checkDataSpace();
		}
		else {
			if(d->createStorage(fObjectId)) {
				stat = true;
			}
			else {
				std::cout<<"\n HBase createDataStorage error";
				stat = false;
			}
		}
/// not a data set?
		//d->close();
		return d;
	}
	
	template<typename T>
	T * openDataStorage(const std::string & name, bool & stat)
	{
		if(!hasNamedData(name.c_str() ) ) {
			stat = false;
			return NULL;
		}
		T * d = new T(name);
		stat = d->openStorage(fObjectId);
		//d->close();		
		//std::cout<<"\n HBase:: open sto end"<<__LINE__<<__FILE__<<"\n";
    //std::cout.flush();
		return d;
	}
	
	void addVertexBlock(const char * nvName, const char * posName, const char * nmlName,
						int * nv, Vector3F * pos, Vector3F * nml);
	void addVertexBlock2(const char * nvName, const char * posName, const char * nmlName, const char * colName,
						int * nv, Vector3F * pos, Vector3F * nml, Vector3F * col);
	
	template<typename T>
	void lsTypedChildHierarchy(std::vector<std::string>& log) 
	{
	    std::vector<std::string > tmNames;
        lsTypedChild<HBase >(tmNames);
        std::vector<std::string>::const_iterator it = tmNames.begin();
        
        for(;it!=tmNames.end();++it) {
            std::string nodeName = *it;
            
            HTransform child(*it);
            
            child.lsTypedChildHierarchy<T>(log);
            
            child.close();
        }
        
        std::vector<std::string > tNames;
        lsTypedChild<T>(tNames);
        std::vector<std::string>::const_iterator ita = tNames.begin();
        
        for(;ita !=tNames.end();++ita) {
            std::string nodeName = *ita;
            log.push_back(nodeName);

        }
        
	}

};

}
#endif        //  #ifndef HBASE_H

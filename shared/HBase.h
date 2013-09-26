/*
 *  HBase.h
 *  masq
 *
 *  Created by jian zhang on 5/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>
class Vector3F;
#include <AllHdf.h>
class HBase : public HGroup {
public:
	HBase(const std::string & path);
	virtual ~HBase();
	
	void addIntAttr(const char * attrName, int *value);
	void addIntData(const char * dataName, unsigned count, int *value);
	void addVector3Data(const char * dataName, unsigned count, Vector3F *value);
	
	void writeIntAttr(const char * attrName, int *value);
	
	char readIntAttr(const char * attrName, int *value);
	char readIntData(const char * dataname, unsigned count, unsigned *dst);
	char readVector3Data(const char * dataname, unsigned count, Vector3F *dst);
	
	std::string fullName(const std::string & partialName) const;
	
	char hasNamedAttr(const char * attrName);
	std::string getAttrName(hid_t attrId);
	
	char hasNamedChild(const char * name);
	std::string getChildName(hsize_t i);
	
	virtual char save();
	virtual char load();
};
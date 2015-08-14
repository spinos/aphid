/*
 *  HAttributeEntry.h
 *  aphid
 *
 *  Created by jian zhang on 8/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HBase.h>
#include <Vector3F.h>

class HAttributeEntry : public HBase {
public:
	enum AttributeType {
		tUnknown = 0,
		tInt = 1,
		tFlt = 2,
		tFlt2 = 3,
		tFlt3 = 4,
		tFlt4 = 5
	};
	
	HAttributeEntry(const std::string & path);
	virtual ~HAttributeEntry();
	
	virtual AttributeType attributeType() const;
	
	virtual char save();
	virtual char load();
    virtual char verifyType();
    
    int savedType() const;
protected:
private:
    int m_savedType;
};

class HIntAttributeEntry : public HAttributeEntry {
public:	
	HIntAttributeEntry(const std::string & path);
	virtual ~HIntAttributeEntry();
	
	virtual AttributeType attributeType() const;
	
	virtual char save(int * src);
	virtual char load(int * dst);
    
protected:
private:
};

class HFltAttributeEntry : public HAttributeEntry {
public:	
	HFltAttributeEntry(const std::string & path);
	virtual ~HFltAttributeEntry();
	
	virtual AttributeType attributeType() const;
	
	virtual char save(float * src);
	virtual char load(float * dst);
    
protected:
private:
};

class HFlt3AttributeEntry : public HAttributeEntry {
public:	
	HFlt3AttributeEntry(const std::string & path);
	virtual ~HFlt3AttributeEntry();
	
	virtual AttributeType attributeType() const;
	
	virtual char save(const Vector3F * src);
	virtual char load(Vector3F * dst);
    
protected:
private:
};


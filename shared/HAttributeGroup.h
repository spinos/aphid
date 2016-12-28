/*
 *  HAttributeGroup.h
 *  aphid
 *
 *  Created by jian zhang on 10/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "HBase.h"
#include <foundation/AAttribute.h>

namespace aphid {

class AAttributeWrap {
public:
	AAttributeWrap();
	virtual ~AAttributeWrap();
	
	AStringAttribute * createString();
	AEnumAttribute * createEnum();
	ACompoundAttribute * createCompound();
	ANumericAttribute * createNumeric(int numericType);
	
	AAttribute * attrib();
protected:

private:
	AAttribute * m_attrib;
};

class HAttributeGroup : public HBase {
public:
	HAttributeGroup(const std::string & path);
	virtual ~HAttributeGroup();
	
	virtual char verifyType();
	virtual char save(AAttribute * data);
	virtual char load(AAttributeWrap & wrap);
protected:
	void writeNumeric(ANumericAttribute * data);
	void writeNumericValueAsInt(ANumericAttribute * data);
	void writeNumericValueAsFlt(ANumericAttribute * data);
	void writeEnum(AEnumAttribute * data);
	void writeString(AStringAttribute * data);
	void writeCompound(ACompoundAttribute * data);
	bool loadNumeric(ANumericAttribute * data);
	bool loadEnum(AEnumAttribute * data);
	bool loadString(AStringAttribute * data);
	bool loadCompound(ACompoundAttribute * data);
	bool readNumericValueAsInt(ANumericAttribute * data);
	bool readNumericValueAsFlt(ANumericAttribute * data);
	
private:
};

}
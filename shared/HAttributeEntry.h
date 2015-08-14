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
protected:
	
private:

};
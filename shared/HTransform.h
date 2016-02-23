/*
 *  HTransform.h
 *  testbcc
 *
 *  Created by jian zhang on 4/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <HBase.h>
namespace aphid {

class BaseTransform;
class HTransform : public HBase {
public:
	HTransform(const std::string & path);
	virtual ~HTransform();
	
	virtual char verifyType();
    virtual char save();
	virtual char save(BaseTransform * tm);
	virtual char load(BaseTransform * tm);
	
private:
	
};

}

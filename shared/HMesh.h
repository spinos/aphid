/*
 *  HMesh.h
 *  masqmaya
 *
 *  Created by jian zhang on 4/13/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HBase.h>
class HMesh : public HBase {
public:
	HMesh(const std::string & path);
	virtual ~HMesh();
	
	char verifyType();
	virtual char save();
	virtual char load();
private:
	
};
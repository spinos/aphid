/*
 *  HMesh.h
 *  masqmaya
 *
 *  Created by jian zhang on 4/13/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseMesh.h>
#include <HBase.h>
class HMesh : public BaseMesh, public HBase {
public:
	HMesh();
	virtual ~HMesh();
	
	static char verifyType(HObject & grp);
	virtual char save();
	virtual char load();
private:
	
};
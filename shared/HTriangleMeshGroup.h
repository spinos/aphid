/*
 *  HTriangleMeshGroup.h
 *  aphid
 *
 *  Created by jian zhang on 7/6/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "HTriangleMesh.h"

namespace aphid {

class ATriangleMeshGroup;
class HTriangleMeshGroup : public HTriangleMesh {
public:
	HTriangleMeshGroup(const std::string & path);
	virtual ~HTriangleMeshGroup();
	
	virtual char verifyType();
	virtual char save(ATriangleMeshGroup * tri);
	virtual char load(ATriangleMeshGroup * tri);
};

}
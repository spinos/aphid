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
class BaseMesh;
class HMesh : public HBase {
public:
	HMesh(const std::string & path);
	virtual ~HMesh();
	
	virtual char verifyType();
	virtual char save(BaseMesh * mesh);
	virtual char load(BaseMesh * mesh);
	
	char saveFaceTag(BaseMesh * mesh, const std::string & tagName, const std::string & dataName);
	char loadFaceTag(BaseMesh * mesh, const std::string & tagName, const std::string & dataName);
private:
	
};
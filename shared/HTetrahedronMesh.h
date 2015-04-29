/*
 *  HTetrahedronMesh.h
 *  testbcc
 *
 *  Created by jian zhang on 4/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <HBase.h>
class BaseBuffer;
class ATetrahedronMesh;

class HTetrahedronMesh : public HBase {
public:
	HTetrahedronMesh(const std::string & path);
	virtual ~HTetrahedronMesh();
	
	char verifyType();
	virtual char save(ATetrahedronMesh * tetra);
	virtual char load(ATetrahedronMesh * tetra);
	
private:
	
};
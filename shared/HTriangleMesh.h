/*
 *  HTriangleMesh.h
 *  testbcc
 *
 *  Created by jian zhang on 4/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <HBase.h>
class ATriangleMesh;

class HTriangleMesh : public HBase {
public:
	HTriangleMesh(const std::string & path);
	virtual ~HTriangleMesh();
	
	virtual char verifyType();
	virtual char save(ATriangleMesh * tri);
	virtual char load(ATriangleMesh * tri);
	
private:
	
};
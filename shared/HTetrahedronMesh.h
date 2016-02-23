#ifndef HTETRAHEDRONMESH_H
#define HTETRAHEDRONMESH_H

/*
 *  HTetrahedronMesh.h
 *  testbcc
 *
 *  Created by jian zhang on 4/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <HBase.h>

namespace aphid {

class ATetrahedronMesh;

class HTetrahedronMesh : public HBase {
public:
	HTetrahedronMesh(const std::string & path);
	virtual ~HTetrahedronMesh();
	
	virtual char verifyType();
	virtual char save(ATetrahedronMesh * tetra);
	virtual char load(ATetrahedronMesh * tetra);

protected:
    char readAftCreation(ATetrahedronMesh * tetra);
private:
	
};

}
#endif        //  #ifndef HTETRAHEDRONMESH_H

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
struct TetrahedronMeshData {
    unsigned m_numTetrahedrons;
    unsigned m_numPoints;
    BaseBuffer * m_anchorBuf;
    BaseBuffer * m_pointBuf;
    BaseBuffer * m_indexBuf;
};
class HTetrahedronMesh : public HBase {
public:
	HTetrahedronMesh(const std::string & path);
	virtual ~HTetrahedronMesh();
	
	char verifyType();
	virtual char save(TetrahedronMeshData * tetra);
	virtual char load(TetrahedronMeshData * tetra);
	
private:
	
};
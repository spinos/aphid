/*
 *  BlockBccMeshBuilder.h
 *  bcc
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AOrientedBox.h>
class CartesianGrid;
class BlockBccMeshBuilder {
public:
	BlockBccMeshBuilder();
	virtual ~BlockBccMeshBuilder();
	
	void build(const AOrientedBox & ob, 
				int gx, int gy, int gz);
protected:
    void addTetrahedron(Vector3F * v);
private:
    CartesianGrid * m_verticesPool;
};
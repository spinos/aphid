#ifndef A_QUAD_MESH_H
#define A_QUAD_MESH_H
/*
 *  AQuadMesh.h
 *  
 *	regular mesh in segment (u, v)
 *  n p (u+1)*(v+1)
 *  n q u*v
 *  Created by jian zhang on 8/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "AGenericMesh.h"

namespace aphid {

class AQuadMesh : public AGenericMesh {
	
	int m_nppu;
public:
	AQuadMesh();
	virtual ~AQuadMesh();
	
	Vector3F * quadP(const int & u, const int & v);
	
	virtual const Type type() const;
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	const unsigned numQuads() const;
	
	void create(const int & useg, const int & vseg);
	unsigned * quadIndices(unsigned idx) const;
	virtual std::string verbosestr() const;
	
protected:
	
private:
	
};

}
#endif

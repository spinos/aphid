#ifndef APH_GENERIC_MESH_H
#define APH_GENERIC_MESH_H

/*
 *  AGenericMesh.h
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <geom/Geometry.h>
#include <Matrix44F.h>
#include <map>

namespace aphid {

class BaseBuffer;
class AGenericMesh : public Geometry {
public:
	AGenericMesh();                                      
	virtual ~AGenericMesh();
	
	virtual const Type type() const;
	virtual const BoundingBox calculateBBox() const;
	
	const unsigned numPoints() const;
	const unsigned numIndices() const;
	Vector3F * points() const;
	Vector3F * vertexNormals() const;
	unsigned * indices() const;
	unsigned * anchors() const;
	
	void copyStripe(AGenericMesh * inmesh, unsigned driftP, unsigned driftI);
	
	const unsigned numAnchoredPoints() const;
	void getAnchorInd(std::map<unsigned, unsigned> & dst) const;
    const Vector3F averageP() const;
    void moveIntoSpace(const Matrix44F & m);
    void clearAnchors();
    BaseBuffer * pointsBuf() const;
protected:
	void createBuffer(unsigned np, unsigned ni);
	void setNumPoints(unsigned n);
	void setNumIndices(unsigned n);
private:
	BaseBuffer * m_points;
	BaseBuffer * m_normals;
	BaseBuffer * m_indices;
	BaseBuffer * m_anchors;
	unsigned m_numPoints, m_numIndices;
};

}
#endif        //  #ifndef AGENERICMESH_H

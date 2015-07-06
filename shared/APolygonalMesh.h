#ifndef APOLYGONALMESH_H
#define APOLYGONALMESH_H

#include "AGenericMesh.h"

class APolygonalMesh : public AGenericMesh {
public:
	APolygonalMesh();
	virtual ~APolygonalMesh();
	
	virtual const Type type() const;
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	const unsigned numPolygons() const;
	
	void create(unsigned np, unsigned ni, unsigned nf);
    void computeFaceDrift();
	unsigned faceCount(unsigned idx) const;
	unsigned * faceCounts() const;
    unsigned * faceDrifts() const;
    
protected:
    unsigned * polygonIndices(unsigned idx) const;
private:
    BaseBuffer * m_faceCounts;
    BaseBuffer * m_faceDrifts;
    unsigned m_numPolygons;
};
#endif        //  #ifndef APOLYGONALMESH_H


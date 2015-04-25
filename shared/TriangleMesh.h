#ifndef TRIANGLEMESH_H
#define TRIANGLEMESH_H

#include "AGenericMesh.h"

class TriangleMesh : public AGenericMesh {
public:
    TriangleMesh();
    virtual ~TriangleMesh();
    virtual const Type type() const;
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	const unsigned numTriangles() const;
	const unsigned numTriangleFaceVertices() const;
private:

};
#endif        //  #ifndef TRIANGLEMESH_H


#pragma once
#include <BaseState.h>
#include <BoundingBox.h>
class BaseBuffer;
class AGenericMesh;
class ADeformer
{
public:
    ADeformer();
    virtual ~ADeformer();
	
	Vector3F * deformedP() const;
	Vector3F * restP() const;

	virtual void setMesh(AGenericMesh * mesh);
	virtual void reset();
	virtual bool solve();
    
	unsigned numVertices() const;
	
	const BoundingBox calculateBBox() const;
	
protected:
	AGenericMesh * mesh();
private:
    AGenericMesh * m_mesh;
	BaseBuffer * m_deformedP;
};

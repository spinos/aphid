#pragma once

#include <BaseMesh.h>

class BaseDeformer
{
public:
    
    BaseDeformer();
    virtual ~BaseDeformer();
	
	virtual void clear();
	
	Vector3F * deformedP();
	Vector3F * getDeformedP() const;
	
	virtual void setMesh(BaseMesh * mesh);
	
	virtual void reset();
    
	virtual char solve();
	
	unsigned numVertices() const;
	
	BoundingBox calculateBBox() const;
	
	void enable();
	void disable();
	bool isEnabled() const;
	
private:
    unsigned m_numVertices;
	Vector3F * m_deformedV;
	Vector3F * m_restV;
	BaseMesh * m_mesh;
	bool m_enabled;
};

#pragma once

#include <BaseMesh.h>

class BaseDeformer
{
public:
    
    BaseDeformer();
    virtual ~BaseDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	
	void reset();
	
	Vector3F * getDeformedData() const;
	
	Vector3F restP(unsigned idx) const;
    
	virtual char solve();
	
	void updateMesh() const;

    unsigned m_numVertices;
	
	Vector3F * m_deformedV;
	Vector3F * m_restV;
	
	BaseMesh * m_mesh;
private:
    
};

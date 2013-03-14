#pragma once

#include <Vector3F.h>
class BaseMesh;
class BaseDeformer
{
public:
    
    BaseDeformer();
    virtual ~BaseDeformer();
	
	virtual void setMesh(BaseMesh * mesh);
	
	Vector3F * getDeformedData() const;
    
	virtual char solve();

    unsigned m_numVertices;
	
	Vector3F * m_deformedV;
	
	BaseMesh * m_mesh;
private:
    
};

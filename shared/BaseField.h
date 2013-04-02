#pragma once

#include <Vector3F.h>
class BaseMesh;
class BaseField
{
public:
    
    BaseField();
    virtual ~BaseField();
	
	virtual void setMesh(BaseMesh * mesh);
	
	Vector3F * getValue() const;
    
	virtual char solve();

    unsigned m_numVertices;
	
	Vector3F * m_value;
	
	BaseMesh * m_mesh;
private:
    
};

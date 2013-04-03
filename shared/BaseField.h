#pragma once

#include <Vector3F.h>
class BaseMesh;
class BaseField
{
public:
    
    BaseField();
    virtual ~BaseField();
	
	virtual void setMesh(BaseMesh * mesh);
	
	Vector3F * getColor() const;
    
	virtual char solve();
	
	void plotColor();

    unsigned m_numVertices;
	
	float * m_value;
	
	BaseMesh * m_mesh;
private:
    Vector3F * m_color;
	
};

#pragma once

#include <Vector3F.h>
#include <map>
class BaseMesh;
class BaseField
{
public:
    
    BaseField();
    virtual ~BaseField();
	
	virtual void setMesh(BaseMesh * mesh);
	void BaseField::addValue(unsigned idx);
	
	Vector3F * getColor() const;
	float * value(unsigned idx);
	float getValue(unsigned setIdx, unsigned valIdx);
	
	virtual char solve();
	
	void plotColor(unsigned idx);

    unsigned m_numVertices;
	unsigned m_activeValue;
	BaseMesh * m_mesh;
private:
    Vector3F * m_color;
	std::map<unsigned, float *> m_values;
};

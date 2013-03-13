#pragma once

#include <Vector3F.h>
class BaseDeformer
{
public:
    
    BaseDeformer();
    virtual ~BaseDeformer();
	void setNumVertices(const unsigned & nv);
	
	Vector3F * getDeformedData() const;
    
	virtual char solve();

    unsigned m_numVertices;
	
	Vector3F * m_deformedV;
private:
    
};

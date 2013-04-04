#pragma once
#include <Anchor.h>
class WeightHandle : public Anchor {
public:
    WeightHandle(SelectionArray & sel);
    
    virtual void translate(Vector3F & dis);
	
	unsigned getVertexIndex() const;
private:
	unsigned m_vertexIdx;
};


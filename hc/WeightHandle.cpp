#include "WeightHandle.h"
#include <Vertex.h>

WeightHandle::WeightHandle(SelectionArray & sel) : Anchor(sel) 
{
	Vertex * v = sel.getVertex(0);
	m_vertexIdx = v->getIndex();
}

void WeightHandle::translate(Vector3F & dis) 
{
    Anchor::translate(dis);
    addWeight(dis.y);
}

unsigned WeightHandle::getVertexIndex() const
{
	return m_vertexIdx;
}

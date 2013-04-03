#include "WeightHandle.h"

WeightHandle::WeightHandle(SelectionArray & sel) : Anchor(sel) {}

void WeightHandle::translate(Vector3F & dis) 
{
    Anchor::translate(dis);
    addWeight(dis.y);
}

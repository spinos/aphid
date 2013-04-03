#pragma once
#include <Anchor.h>
class WeightHandle : public Anchor {
public:
    WeightHandle(SelectionArray & sel);
    
    virtual void translate(Vector3F & dis);
private:

};


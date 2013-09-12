#pragma once
#include <AllMath.h>
class MlFeather {
public:
    MlFeather();
    virtual ~MlFeather();
    void createNumSegment(short x);
    
    float * quilly();
    float * getQuilly() const;
    Vector2F * vaneAt(short seg, short side);
    Vector2F * getVaneAt(short seg, short side) const;
private:
    float *m_quilly;
    Vector2F * m_vaneVertices;
    short m_numSeg;
};

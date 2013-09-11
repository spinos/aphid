#pragma once

class MlFeather {
public:
    MlFeather();
    virtual ~MlFeather();
    void setNumSegment(short x);
private:
    float *m_quillz;
    short m_numSeg;
};

#ifndef APOLYGONALUV_H
#define APOLYGONALUV_H

class BaseBuffer;
class APolygonalUV {
public:
    APolygonalUV();
    virtual ~APolygonalUV();
    
    void create(unsigned ncoords, unsigned ninds);
    
    float * ucoord() const;
    float * vcoord() const;
    unsigned * indices() const;
    const unsigned numCoords() const;
    const unsigned numIndices() const;
protected:

private:
    BaseBuffer * m_ucoord;
    BaseBuffer * m_vcoord;
    BaseBuffer * m_indices;
    unsigned m_numCoords, m_numIndices;
};
#endif        //  #ifndef APOLYGONALUV_H


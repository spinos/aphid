#ifndef CSRMATRIX_H
#define CSRMATRIX_H
#include <map>
typedef std::map<unsigned, unsigned> CSRMap;
class BaseBuffer;
class CSRMatrix 
{
public:
    enum ValueType{
        tFloat = 4,
        tVec3 = 12,
        tMat33 = 36,
        tMat44 = 64
    };
    
    CSRMatrix();
    virtual ~CSRMatrix();
    
    void create(ValueType type, unsigned m, const CSRMap & elms);
    
    void * value();
    void * rowValue(unsigned i);
    unsigned * rowPtr();
    unsigned * colInd();
    
    const unsigned dimension() const;
    const unsigned numNonZero() const;
    const ValueType valueType() const;
    
    void verbose();
protected:

private:
    BaseBuffer * m_value;
    BaseBuffer * m_rowPtr;
    BaseBuffer * m_colInd;
    unsigned m_dimension, m_numNonZero;
    ValueType m_valType;
};
#endif        //  #ifndef CSRMATRIX_H


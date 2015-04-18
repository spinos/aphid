// http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-indexing-and-data-formats

#include "CSRMatrix.h"
#include "BaseBuffer.h"
#include <iostream>
CSRMatrix::CSRMatrix() 
{
    m_value = new BaseBuffer;
    m_rowPtr = new BaseBuffer;
    m_colInd = new BaseBuffer;
}

CSRMatrix::~CSRMatrix() 
{
    delete m_value;
    delete m_rowPtr;
    delete m_colInd;
}

void CSRMatrix::create(CSRMatrix::ValueType type, unsigned m, const CSRMap & elms)
{
    m_valType = type;
    m_dimension = m;
    unsigned nnz = elms.size();
    m_numNonZero = nnz;
    m_value->create(nnz * type);
    m_rowPtr->create((m+1) * 4);
    m_colInd->create(nnz * 4);
    
    unsigned * I = rowPtr();
    unsigned * J = colInd();
    
    unsigned lastRow = m+2;
    unsigned row, col;
    unsigned ielm = 0;
    CSRMap::const_iterator it = elms.begin();
    for(; it !=elms.end(); ++it) {
        row = it->first/m;
        col = it->first%m;
        
        if(row != lastRow) {
            I[row] = ielm;
            lastRow = row;
        }
        
        J[it->second] = col;
        
        ielm++;
    }
    I[m] = ielm;
}

void * CSRMatrix::value()
{ return m_value->data(); }

void * CSRMatrix::rowValue(unsigned i)
{
    char * d = (char *)m_value->data();
    return &d[rowPtr()[i] * m_valType];
}

unsigned * CSRMatrix::rowPtr()
{ return (unsigned *)m_rowPtr->data(); }

unsigned * CSRMatrix::colInd()
{ return (unsigned *)m_colInd->data(); }

const unsigned CSRMatrix::dimension() const
{ return m_dimension; }

const unsigned CSRMatrix::numNonZero() const
{ return m_numNonZero; }

const CSRMatrix::ValueType CSRMatrix::valueType() const
{ return m_valType; }

unsigned CSRMatrix::maxNNZRow()
{
    unsigned i;
    unsigned * row = rowPtr();
    unsigned mnnzpr = 0;
    for(i = 1; i<=m_dimension; i++) {
        if(row[i]-row[i-1] > mnnzpr)
            mnnzpr = row[i]-row[i-1];
    }
    return mnnzpr;
}

void CSRMatrix::verbose()
{
    unsigned lastRow = m_dimension + 2;
    unsigned i, j;
    unsigned * row = rowPtr();
    j = 0;
    for(i=0; i< m_numNonZero; i++) {
        if(i == *row) {
            std::cout<<"\n row"<<j<<" ("<<*row<<") ";
            j++;
            row++;
        }
        std::cout<<" "<<colInd()[i];
    }
    std::cout<<"\nn non-zero "<<*row<<"\n";
    std::cout<<"dimension "<<m_dimension<<"\n";
    std::cout<<"value size "<<m_valType<<"\n";
}


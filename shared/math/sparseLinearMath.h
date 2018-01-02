#ifndef APH_SPARSE_LINEARMATH_H
#define APH_SPARSE_LINEARMATH_H

#include "linearMath.h"

namespace aphid {
    
/// dynamic sparse vector can add/remove non-zero element
/// most elements are zero, non-zero elements limited to MNNZ
/// element index [0, MNNZ-1] to sparse index

template<typename T, int MNNZ>
class SparseVector {

    T m_v[MNNZ];
    int m_elmIndex[MNNZ];
    int m_numElms;
    
public:
    SparseVector();
    virtual ~SparseVector();
/// i-th element    
     T operator[](int i);
     
private:
    struct SearchResult {
        int _found;
        int _followedBy;
    };
/// to i-th element
/// -1 if not found
    SearchResult indexTo(int i) const;
/// between first and last
    void searchIndex(int i, SearchResult* sr) const;
    
};

template<typename T, int MNNZ>
SparseVector<T, MNNZ>::SparseVector() : m_numElms(0) 
{
    memset(m_elmIndex, 0, MNNZ * 4);
}

template<typename T, int MNNZ>
SparseVector<T, MNNZ>::~SparseVector()
{}

template<typename T, int MNNZ>
T SparseVector<T, MNNZ>::operator[](int i)
{
    SearchResult sr = indexTo(i);
    if(sr._found < 0) {
        return T(0); 
    }
    return m_v[sr._found];
}

template<typename T, int MNNZ>
SparseVector<T, MNNZ>::SearchResult SparseVector<T, MNNZ>::indexTo(int i) const
{
    SearchResult sr;
    sr._found = -1;
    if(m_numElms == 0) {
        sr._followedBy = 0;
    }
    if(m_numElms == 1) {
        sr._found = 0;
        sr._followedBy = 1;
    }
    if(i < m_elmIndex[0]) {
        sr._followedBy = 0;
    }
    if(i > m_elmIndex[m_numElms-1]) {
/// overflow
        sr._followedBy = m_numElms + 1;
    }
    searchIndex(i, 0, m_numElms-1, &sr);
    return sr;
}

template<typename T, int MNNZ>
void SparseVector<T, MNNZ>::searchIndex(int i, SearchResult* sr) const
{
    int low, high, mid;
    low = 0;
    high = m_numElms-1;
    while(low < high) {
        if(i == m_elmIndex[low]) {
            sr._found = low;
            sr._followedBy = low + 1;
            return;
        }
        if(i == m_elmIndex[high]) {
            sr._found = high;
            sr._followedBy = high + 1;
            return; 
        }
        mid = (low + high) / 2;
        if(i == m_elmIndex[mid]) {
            sr._found = mid;
            sr._followedBy = mid + 1;
            return;  
        }
        if(i < m_elmIndex[mid]) {
            high = mid;
        } else {
            low = mid;
        }
    }
    sr._followedBy = mid + 1;
    return -1;
}

/// column-major sparse matrix in csr

template<typename T>
class SparseMatrix {

    T * m_v;
    int * m_rowIndices;
    int * m_columnBegins;
    int m_numColumns;
    int m_numRows;
    int m_numMaxNonZero;
    
public:
    SparseMatrix();
    virtual ~SparseMatrix();
    
    void create(int numRow, int numCol, int maxNz);
    int numColumns() const;
    int numRows() const;
    int maxNumNonZero() const;
    
protected:

private:
    void clear();
};

template<typename T>
SparseMatrix<T>::SparseMatrix() : m_v(NULL), m_rowIndices(NULL), m_columnBegins(NULL), m_numMaxNonZero(0) {}

template<typename T>
SparseMatrix<T>::~SparseMatrix()
{ clear(); }

template<typename T>
void SparseMatrix<T>::create(int numRow, int numCol, int maxNz)
{
    clear();
    m_numColumns = numCol;
    m_numRows = numRow;
    m_numMaxNonZero = maxNz;
    m_v = new T[maxNz];
    m_rowIndices = new int[maxNz];
    m_columnBegins = new int[numCol+1];
}

template<typename T>
int SparseMatrix<T>::numColumns() const
{ return m_numColumns; }

template<typename T>
int SparseMatrix<T>::numRows() const
{ return m_numRows; }

template<typename T>
int SparseMatrix<T>::maxNumNonZero() const
{ return m_numMaxNonZero; }

template<typename T>
void SparseMatrix<T>::clear() 
{ 
    if(m_v) {
        delete[] m_v;
        m_v = NULL;
    }
    if(m_rowIndices) {
        delete[] m_rowIndices;
        m_rowIndices = NULL;
    }
    if(m_columnBegins) {
        delete[] m_columnBegins;
        m_columnBegins = NULL;
    }
    m_numColumns = 0;
    m_numRows = 0;
    m_numMaxNonZero = 0;
}

}

#endif


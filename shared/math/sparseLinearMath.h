#ifndef APH_SPARSE_LINEARMATH_H
#define APH_SPARSE_LINEARMATH_H

#include "linearMath.h"

namespace aphid {

struct SparseSearchResult {
	int _found;
	int _followedBy;
};

template<typename T>
class SparseIterator {
	
	int m_p;
	int* m_index;
	T* m_v;
	
public:
	SparseIterator(int p, int* i, T* x) : m_p(p),
	m_index(i),
	m_v(x)
	{}
	
	SparseIterator& operator++() 
	{
		m_p++;
		return *this;
	}
	
	bool operator==(SparseIterator other) const 
	{
		return m_p == other.m_p;
	}
	
	bool operator!=(SparseIterator other) const 
	{
		return !(*this == other);
	}
	
	int index() const
	{
		return m_index[m_p];
	}
	
	const T& value() const
	{
		return m_v[m_p];
	}
	
};
    
/// dynamic sparse vector can add/remove non-zero element
/// presumably most elements are zero
/// capacity to store non-zero elements can be extended by 32
/// element index [0, NNZ-1] to sparse index

template<typename T>
class SparseVector {

    T* m_v;
    int* m_elmIndex;
    int m_numElms;
	int m_cap;
	    
public:
    SparseVector();
    virtual ~SparseVector();
	
	const T& valueByPhysicalIndex(int i) const;
	T& valueByPhysicalIndex(int i);
	
/// i-th element    
	T operator[](int i) const;
/// insert if not found i-th element
	void set(int i, const T& val);
/// a_i <- a_i + b
	void add(int i, const T& b);
	void remove(int i);
	
	SparseIterator<T> begin();
	SparseIterator<T> end();
	
private:
    void expand();
/// to i-th element
/// -1 if not found
    SparseSearchResult indexTo(int i) const;
/// between first and last
    void searchIndex(int i, SparseSearchResult* sr) const;
/// index[physicalIndex] <- i
/// v[physicalIndex] <- val
	void insertElement(int i, const T& val, int physicalIndex);
	
};

template<typename T>
SparseVector<T>::SparseVector() : m_numElms(0), m_cap(32)
{
	m_v = new T[m_cap];
	m_elmIndex = new int[m_cap];
    memset(m_elmIndex, 0, m_cap * 4);
}

template<typename T>
SparseVector<T>::~SparseVector()
{
	delete[] m_v;
	delete[] m_elmIndex;
}

template<typename T>
const T& SparseVector<T>::valueByPhysicalIndex(int i) const
{
	return m_v[i];
}

template<typename T>
T& SparseVector<T>::valueByPhysicalIndex(int i)
{
	return m_v[i];
}

template<typename T>
T SparseVector<T>::operator[](int i) const
{
    SparseSearchResult sr = indexTo(i);
    if(sr._found < 0) {
        return T(0); 
    }
    return m_v[sr._found];
}

template<typename T>
void SparseVector<T>::set(int i, const T& val)
{
	SparseSearchResult sr = indexTo(i);
	if(sr._found > -1) {
		m_v[sr._found] = val;
		return;
	}
	
	insertElement(i, val, sr._followedBy);
}

template<typename T>
void SparseVector<T>::add(int i, const T& b)
{
	SparseSearchResult sr = indexTo(i);
	if(sr._found > -1) {
		m_v[sr._found] += b;
		return;
	}
	
	insertElement(i, b, sr._followedBy);
}

template<typename T>
void SparseVector<T>::remove(int i)
{
	SparseSearchResult sr = indexTo(i);
	if(sr._found < 0) {
		return;
	}
	
	for(int j = sr._found;j<m_numElms-1;++j) {
		m_elmIndex[j] = m_elmIndex[j+1];
		m_v[j] = m_v[j+1];
	}
	m_numElms--;
}

template<typename T>
void SparseVector<T>::expand()
{
	T* buf = new T[m_cap];
	memcpy(buf, m_v, m_cap * sizeof(T) );
	delete[] m_v;
	
	int* ibuf = new int[m_cap];
	memcpy(ibuf, m_elmIndex, m_cap * 4 );
	delete[] m_elmIndex;
	
	m_cap += 32;
	m_v = new T[m_cap];
	m_elmIndex = new int[m_cap];
	
	memcpy(m_v, buf, m_cap * sizeof(T) );
	memcpy(m_elmIndex, ibuf, m_cap * 4 );
	
	delete[] buf;
	delete[] ibuf;
	
}

template<typename T>
void SparseVector<T>::insertElement(int i, const T& val, int physicalIndex)
{
	if(m_numElms >= m_cap) {
		expand();
	}
	
	if(m_numElms == 0) {
/// first
		m_elmIndex[0] = i;
		m_v[0] = val;
		m_numElms = 1;
		return;
	}
	
	if(physicalIndex == m_numElms) {
/// last
		m_elmIndex[physicalIndex] = i;
		m_v[physicalIndex] = val;
		m_numElms += 1;
		return;
	}
	
	for(int j=m_numElms; j>physicalIndex;--j) {
		m_elmIndex[j] = m_elmIndex[j-1];
		m_v[j] = m_v[j-1];
	}
	
	m_elmIndex[physicalIndex] = i;
	m_v[physicalIndex] = val;
	m_numElms += 1;
}

template<typename T>
SparseSearchResult SparseVector<T>::indexTo(int i) const
{
    SparseSearchResult sr;
    sr._found = -1;
    if(m_numElms == 0) {
/// empty
        sr._followedBy = 0;
		return sr;
    }
    if(i < m_elmIndex[0]) {
/// low bound
        sr._followedBy = 0;
		return sr;
    }
    if(i > m_elmIndex[m_numElms-1]) {
/// high bound
        sr._followedBy = m_numElms;
		return sr;
    }
    searchIndex(i, &sr);
    return sr;
}

template<typename T>
void SparseVector<T>::searchIndex(int i, SparseSearchResult* sr) const
{	
    int low, high, mid;
    low = 0;
    high = m_numElms-1;
    for(;;) {
        if(i == m_elmIndex[low]) {
            sr->_found = low;
            sr->_followedBy = low + 1;
            return;
        }
        if(i == m_elmIndex[high]) {
            sr->_found = high;
            sr->_followedBy = high + 1;
            return; 
        }
        mid = (low + high) / 2;
        if(i == m_elmIndex[mid]) {
            sr->_found = mid;
            sr->_followedBy = mid + 1;
            return;  
        }
        if(i < m_elmIndex[mid]) {
            high = mid;
        } else {
            low = mid;
        }
		sr->_followedBy = high;
		if(low+1 >= high) {
/// no more
			break;
		}
    }

}

template<typename T>
SparseIterator<T> SparseVector<T>::begin()
{
	return SparseIterator<T>(0, m_elmIndex, m_v);
}

template<typename T>
SparseIterator<T> SparseVector<T>::end()
{
	return SparseIterator<T>(m_numElms, m_elmIndex, m_v);
}

/// dynamic sparse matrix add/remove non-zero element
/// if ncol > nrow, store in rows
/// each column/row is a sparse vector
/// all columns/row have non-zero elements

template<typename T>
class SparseMatrix {
	
typedef SparseVector<T> SVecTyp;
	SVecTyp* m_vecs;
	int m_numCols;
    int m_numRows;
	
	enum CompressFormat {
		cfUnknown = 0,
		cfColumnMajor,
		cfRowMajor,
	};
	
	CompressFormat m_format;
	
public:
	SparseMatrix();
	virtual ~SparseMatrix();
	
/// nrow-by-ncol
	void create(int nrow, int ncol);
/// i-th row j-th column
	void set(int i, int j, const T& val);
/// a_ij <- a_ij + b
	void add(int i, int j, const T& b);
	
	T get(int i, int j) const;
	
	const int& numCols() const;
    const int& numRows() const;
	
	SparseMatrix transposed() const;
/// c <- ab
	SparseMatrix operator*(const SparseMatrix& b) const;
/// i-th column
	SVecTyp& row(int i) const;
/// j-th column
	SVecTyp& column(int j) const;
	
	bool isColumnMajor() const;
/// test
	void printMatrix() const;
	
private:
	void clear();
	
};

template<typename T>
SparseMatrix<T>::SparseMatrix() :
m_vecs(NULL),
m_numCols(0),
m_numRows(0), m_format(cfUnknown)
{}

template<typename T>
SparseMatrix<T>::~SparseMatrix()
{
	clear();
}

template<typename T>
void SparseMatrix<T>::clear()
{
	if(m_vecs)
		delete[] m_vecs;
}

template<typename T>
void SparseMatrix<T>::create(int nrow, int ncol)
{
	clear();
	m_format = (ncol > nrow) ? cfRowMajor: m_format = cfColumnMajor;
	if(m_format == cfRowMajor) {
		m_vecs = new SVecTyp[nrow];
	} else {
		m_vecs = new SVecTyp[ncol];
	}
	m_numRows = nrow;
	m_numCols = ncol;
}

template<typename T>
void SparseMatrix<T>::set(int i, int j, const T& val)
{
	if(m_format == cfRowMajor) {
		m_vecs[i].set(j, val);
	} else {
		m_vecs[j].set(i, val);
	}
}

template<typename T>
void SparseMatrix<T>::add(int i, int j, const T& b)
{
	if(m_format == cfRowMajor) {
		m_vecs[i].add(j, b);
	} else {
		m_vecs[j].add(i, b);
	}
}

template<typename T>
T SparseMatrix<T>::get(int i, int j) const
{
	if(m_format == cfRowMajor) {
		return m_vecs[i][j];
	}
	return m_vecs[j][i];
}

template<typename T>
const int& SparseMatrix<T>::numCols() const
{
	return m_numCols;
}

template<typename T>
const int& SparseMatrix<T>::numRows() const
{
	return m_numRows;
}

template<typename T>
SparseMatrix<T> SparseMatrix<T>::transposed() const
{
	SparseMatrix<T> tm;
	tm.create(numCols(), numRows() );
	if(m_format == cfRowMajor) {
		for(int i=0;i<numRows();++i) {
/// i-th row
			SparseIterator<int> iter = m_vecs[i].begin();
			SparseIterator<int> itEnd = m_vecs[i].end();
			for(;iter != itEnd;++iter) {
				tm.set(iter.index(), i, iter.value() );
			}
		}
	} else {
		for(int j=0;j<numCols();++j) {
/// j-th column
			SparseIterator<int> iter = m_vecs[j].begin();
			SparseIterator<int> itEnd = m_vecs[j].end();
			for(;iter != itEnd;++iter) {
				tm.set(j, iter.index(), iter.value() );
			}
		}
	}
	return tm;
}

template<typename T>
SparseMatrix<T> SparseMatrix<T>::operator*(const SparseMatrix& b) const
{
	SparseMatrix<T> c;
	c.create(numRows(), b.numCols() );
/// ijk-form matrix-matrix product
	if(m_format == cfRowMajor) {
/// b is column-major
		for(int i=0;i<numRows();++i) {
/// i-th row in a
			SVecTyp& ai = row(i);
			for(int j=0;j<b.numCols();++j) {
				T cij = 0;

				SparseIterator<T> iter = ai.begin();
				SparseIterator<T> itEnd = ai.end();
				for(;iter != itEnd;++iter) {
/// i-th column in b
					cij += iter.value() * b.get(iter.index(), j);
				}
				if(cij != 0) {
					c.set(i, j, cij);
				}
			}
		}
	} else {
/// a is column-major, b can be column-major or row-major
		for(int i=0;i<numRows();++i) {
			for(int j=0;j<b.numCols();++j) {
				T cij = 0;
				
				if(b.isColumnMajor() ) {
/// j-th column in b
					SVecTyp& bj = b.column(j);
					SparseIterator<T> iter = bj.begin();
					SparseIterator<T> itEnd = bj.end();
					for(;iter != itEnd;++iter) {
/// i-th row in a
						cij += get(i, iter.index() ) * iter.value();
					}
				} else {
					for(int k=0;k<numCols();++k) {
						cij += get(i, k) * b.get(k, j);
					}
				}
				if(cij != 0) {
					c.set(i, j, cij);
				}
			}
		}
	}
	return c;
}

template<typename T>
bool SparseMatrix<T>::isColumnMajor() const
{ return m_format == cfColumnMajor; }

template<typename T>
SparseVector<T>& SparseMatrix<T>::column(int j) const
{
	return m_vecs[j];
}

template<typename T>
SparseVector<T>& SparseMatrix<T>::row(int i) const
{
	return m_vecs[i];
}

template<typename T>
void SparseMatrix<T>::printMatrix() const
{
	std::cout<<"\n "<<m_numRows<<"-by-"<<m_numCols<<" mat";
	if(isColumnMajor() ) {
		std::cout<<" column-major\n";
	} else {
		std::cout<<" row-major\n";
	}
	for(int i=0;i<m_numRows;++i) {
		for(int j=0;j<m_numCols;++j) {
			T e = get(i,j);
			std::cout<<" "<<e;
		}
		std::cout<<"\n";
	}
	std::cout.flush();
}

/// sparse matrix in Compressed Sparse Row (csr) format
/// http://www.cs.colostate.edu/~mcrob/toolbox/c++/sparseMatrix/sparse_matrix_compression.html

template<typename T>
class CSRMatrix {

    T * m_v;
    int * m_rowIndices;
    int * m_columnBegins;
    int m_numColumns;
    int m_numRows;
    int m_numMaxNonZero;
    
public:
    CSRMatrix();
    virtual ~CSRMatrix();
    
    void create(int numRow, int numCol, int maxNz);
    int numColumns() const;
    int numRows() const;
    int maxNumNonZero() const;
    
protected:

private:
    void clear();
};

template<typename T>
CSRMatrix<T>::CSRMatrix() : m_v(NULL), m_rowIndices(NULL), m_columnBegins(NULL), m_numMaxNonZero(0) {}

template<typename T>
CSRMatrix<T>::~CSRMatrix()
{ clear(); }

template<typename T>
void CSRMatrix<T>::create(int numRow, int numCol, int maxNz)
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
int CSRMatrix<T>::numColumns() const
{ return m_numColumns; }

template<typename T>
int CSRMatrix<T>::numRows() const
{ return m_numRows; }

template<typename T>
int CSRMatrix<T>::maxNumNonZero() const
{ return m_numMaxNonZero; }

template<typename T>
void CSRMatrix<T>::clear() 
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


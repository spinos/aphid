#ifndef LINEARMATH_H
#define LINEARMATH_H

#include <iostream>
#include <sstream>
#include <cmath>
#include "cblasTempl.h"

// MIN, MAX macros
#define MIN(a,b) (((a) > (b)) ? (b) : (a))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define SIGN(a) (((a) < 0) ? -1.0 : 1.0)
#define ABS(a) (((a) < 0) ? -(a) : (a))

namespace lfr {
template<typename T>
class DenseVector {
   
    T * m_v;
    int m_numElements;
	bool m_isReferenced;
    
public:
    DenseVector();             
	DenseVector(T * v, int n);
    virtual ~DenseVector();
    
    void create(int n);
    int numElements() const;
    
    T& operator()(const int i);
    T operator()(const int i) const;
    
    T* v() const;
	T* raw();
    
    void setZero();
/// ||x||
	T norm() const;
	void scale(const T s);
    void normalize();
/// element index of max value
	int maxInd() const;
/// max element value
	T max() const;
/// element index of max value
	int maxAbsInd() const;
/// max element value
	T maxAbs() const;
	
	void add(const DenseVector<T> & x);
	void minus(const DenseVector<T> & x);
	void copy(const DenseVector<T> & x);
	
	friend std::ostream& operator<<(std::ostream &output, const DenseVector<T> & p) {
        output << p.str();
        return output;
    }
protected:
    
private:
	const std::string str() const;
    void clear();
    
};

template<typename T>
DenseVector<T>::DenseVector() : m_v(NULL), m_numElements(0), m_isReferenced(false) {}

template<typename T>
DenseVector<T>::DenseVector(T * v, int n) : m_v(v), m_numElements(n), m_isReferenced(true) {}

template<typename T>
DenseVector<T>::~DenseVector()
{ clear(); }

template<typename T>
void DenseVector<T>::create(int n)
{
    clear();
    m_v = new T[n];
    m_numElements = n;
}

template<typename T>
int DenseVector<T>::numElements() const
{ return m_numElements; }

template<typename T>
T& DenseVector<T>::operator()(const int i)
{ return m_v[i]; }

template<typename T>
T DenseVector<T>::operator()(const int i) const
{ return m_v[i]; }

template<typename T>
T* DenseVector<T>::v() const
{ return m_v; }

template<typename T>
T* DenseVector<T>::raw()
{ return m_v; }

template<typename T>
void DenseVector<T>::setZero()
{ memset(m_v,0,m_numElements*sizeof(T)); }

template<typename T>
T DenseVector<T>::norm() const
{
	T s = 0.f;
	for(int i = 0; i<m_numElements; i++) s += m_v[i] * m_v[i];
	return sqrt(s);
}

template<typename T>
void DenseVector<T>::scale(const T s)
{
	for(int i = 0; i<m_numElements; i++) m_v[i] *= s;
}

template<typename T>
void DenseVector<T>::normalize()
{
	const T s = norm();
	if(s > 1e-9) scale(1.0 / s);
}

template<typename T>
int DenseVector<T>::maxInd() const
{
	int imax = 0;
	T vmax = m_v[0];
	for(int i=1; i<m_numElements; i++) {
		T cur = m_v[i];
		if(cur > vmax) {
			imax = i;
			vmax = cur;
		}
	}
	return imax;
}

template<typename T>
T DenseVector<T>::max() const
{ return m_v[maxInd()]; }

template<typename T>
int DenseVector<T>::maxAbsInd() const
{
	int imax = 0;
	T vmax = ABS(m_v[0]);
	for(int i=1; i<m_numElements; i++) {
		T cur = ABS(m_v[i]);
		if(cur > vmax) {
			imax = i;
			vmax = cur;
		}
	}
	return imax;
}

template<typename T>
T DenseVector<T>::maxAbs() const
{ return m_v[maxAbsInd()]; }

template<typename T>
void DenseVector<T>::add(const DenseVector<T> & x)
{
	cblas_axpy<T>(m_numElements, T(1.0), x.v(), 1, m_v, 1);
}

template<typename T>
void DenseVector<T>::minus(const DenseVector<T> & x)
{
	cblas_axpy<T>(m_numElements, T(-1.0), x.v(), 1, m_v, 1);
}

template<typename T>
void DenseVector<T>::copy(const DenseVector<T> & x)
{
	create(x.numElements());
	memcpy(m_v, x.v(), m_numElements*sizeof(T));
}

template<typename T>
const std::string DenseVector<T>::str() const
{
	std::stringstream sst;
	sst<<m_numElements<<" vector \n|";
	for (int i = 0; i<m_numElements; ++i) {
	  sst<<" "<<static_cast<double>(m_v[i]);
   }
   sst<<" |\n";
   return sst.str();
}

template<typename T>
void DenseVector<T>::clear()
{
    m_numElements = 0;
	if(m_isReferenced) return;
	if(m_v) {
        delete[] m_v;
        m_v = NULL;
    }
}

/// column-major dense matrix

template<typename T>
class DenseMatrix {
	friend class DenseVector<T>;
    T * m_v;
    int m_numColumns;
    int m_numRows;
    
public:
    DenseMatrix();
    virtual ~DenseMatrix();
    
    void create(int numCol, int numRow);
    int numColumns() const;
    int numRows() const;
	
/// i is column index, j is row index
    T& operator()(const int i, const int j);
    T operator()(const int i, const int j) const;
	T* column(const int i) const;
	void getColumn(DenseVector<T> & x, const int i) const;
	
	void setZero();
	void scale(const T s);
	
/// normalize each column
	void normalize();
/// AT * A
	void AtA(DenseMatrix<T>& dst) const;
/// aii += diag
	void addDiagonal(const T diag);
/// copy upper-right part to lower-left part
	void fillSymmetric();
/// b = alpha A * x + beta b
	void mult(DenseVector<T>& b, const DenseVector<T>& x, 
            const T alpha = 1.0, const T beta = 0.0) const;
/// b = alpha AT * x + beta b
	void multTrans(DenseVector<T>& b, const DenseVector<T>& x, 
            const T alpha = 1.0, const T beta = 0.0) const;
	
	friend std::ostream& operator<<(std::ostream &output, const DenseMatrix<T> & p) {
        output << p.str();
        return output;
    }

protected:

private:
	const std::string str() const;
    void clear();
    
};

template<typename T>
DenseMatrix<T>::DenseMatrix():m_v(NULL), m_numColumns(0), m_numRows(0) {}

template<typename T>
DenseMatrix<T>::~DenseMatrix() 
{ clear(); }

template<typename T>
void DenseMatrix<T>::create(int numCol, int numRow)
{
    clear();
    m_numColumns = numCol;
    m_numRows = numRow;
    m_v = new T[numCol*numRow];
}

template<typename T>
int DenseMatrix<T>::numColumns() const
{ return m_numColumns; }

template<typename T>
int DenseMatrix<T>::numRows() const
{ return m_numRows; }

template <typename T> 
T& DenseMatrix<T>::operator()(const int i, const int j) 
{ return m_v[i*m_numRows+j]; }

template <typename T> 
T DenseMatrix<T>::operator()(const int i, const int j) const 
{ return m_v[i*m_numRows+j]; }

template <typename T>
T* DenseMatrix<T>::column(const int i) const
{ return &m_v[i*m_numRows]; }

template <typename T>
void DenseMatrix<T>::getColumn(DenseVector<T> & x, const int i) const
{
	memcpy(x.raw(), column(i), m_numRows*sizeof(T));
}

template <typename T> 
void DenseMatrix<T>::setZero()
{
	int i = 0;
	for(;i<m_numColumns;i++) {
		DenseVector<T> d(&m_v[i*m_numRows], m_numRows);
		d.setZero();
	}
}

template <typename T> 
void DenseMatrix<T>::scale(const T s)
{
	int i = 0;
	for(;i<m_numColumns;i++) {
		DenseVector<T> d(&m_v[i*m_numRows], m_numRows);
		d.scale(s);
	}
}

template <typename T> 
void DenseMatrix<T>::normalize()
{
	int i = 0;
	for(;i<m_numColumns;i++) {
		DenseVector<T> d(&m_v[i*m_numRows], m_numRows);
		d.normalize();
	}
}

template <typename T>
void DenseMatrix<T>::AtA(DenseMatrix<T>& dst) const 
{
/// syrk performs a rank-n update of an n-by-n symmetric matrix c, that is:
/// c := alpha*a'*a + beta*c
/// a is k-by-n matrix
/// c is n-by-n matrix
/// alpha = 1, beta = 0 
	dst.create(m_numColumns, m_numColumns);
	cblas_syrk<T>(CblasColMajor, CblasUpper, CblasTrans, m_numColumns, m_numRows, 
										T(1.0), m_v, m_numRows, 
										T(0.0), dst.m_v, m_numColumns);
    dst.fillSymmetric();
}

template <typename T>
void DenseMatrix<T>::mult(DenseVector<T>& b, const DenseVector<T>& x, 
            const T alpha, const T beta) const
{
	cblas_gemv<T>(CblasColMajor, CblasNoTrans, m_numRows, m_numColumns, 
							alpha, m_v, m_numRows, 
							x.v(), 1, 
							beta, b.v(), 1);
}

template <typename T>
void DenseMatrix<T>::multTrans(DenseVector<T>& b, const DenseVector<T>& x, 
            const T alpha, const T beta) const
{
	cblas_gemv<T>(CblasColMajor, CblasTrans, m_numRows, m_numColumns, 
							alpha, m_v, m_numRows, 
							x.v(), 1, 
							beta, b.v(), 1);
}

template <typename T>
void DenseMatrix<T>::fillSymmetric() 
{
	for (int i = 0; i<m_numColumns; ++i) {
      for (int j =0; j<i; ++j) {
         m_v[j*m_numRows+i]=m_v[i*m_numRows+j];
      }
   }
}

template <typename T>  
void DenseMatrix<T>::addDiagonal(const T diag) 
{ 
	const int n = MIN(m_numRows, m_numColumns);
	for(int i = 0; i<n; ++i) 
		m_v[i*m_numRows+i] += diag; 
};

template<typename T>
const std::string DenseMatrix<T>::str() const
{
	std::stringstream sst;
	sst<<m_numRows<<"-by-"<<m_numColumns<<" matrix ";
	for (int i = 0; i<m_numRows; ++i) {
      sst<<"\n|";
	  for (int j = 0; j<m_numColumns; ++j) {
         sst<<" "<<static_cast<double>(m_v[j*m_numRows+i]);
      }
      sst<<" |";
   }
   sst<<"\n";
   return sst.str();
}

template<typename T>
void DenseMatrix<T>::clear() 
{ 
    if(m_v) {
        delete[] m_v;
        m_v = NULL;
    }
    m_numColumns = 0;
    m_numRows = 0;
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
    
    void create(int numCol, int numRow, int maxNz);
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
void SparseMatrix<T>::create(int numCol, int numRow, int maxNz)
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

} /// end of namespace lfr

#endif        //  #ifndef LINEARMATH_H


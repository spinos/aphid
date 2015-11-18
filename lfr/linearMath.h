#ifndef LINEARMATH_H
#define LINEARMATH_H

#include <iostream>
#include <sstream>
#include <cmath>
#include "clapackTempl.h"

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
	T maxVal() const;
/// element index of max value
	int maxAbsInd() const;
/// max element value
	T maxAbsVal() const;
	
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
T DenseVector<T>::maxVal() const
{ return m_v[maxInd()]; }

template<typename T>
int DenseVector<T>::maxAbsInd() const
{
	int imax = 0;
	T vmax = abs(m_v[0]);
	for(int i=1; i<m_numElements; i++) {
		T cur = abs(m_v[i]);
		if(cur > vmax) {
			imax = i;
			vmax = cur;
		}
	}
	return imax;
}

template<typename T>
T DenseVector<T>::maxAbsVal() const
{ return m_v[maxAbsInd()]; }

template<typename T>
void DenseVector<T>::add(const DenseVector<T> & x)
{
	clapack_axpy<T>(m_numElements, T(1.0), x.v(), 1, m_v, 1);
}

template<typename T>
void DenseVector<T>::minus(const DenseVector<T> & x)
{
	clapack_axpy<T>(m_numElements, T(-1.0), x.v(), 1, m_v, 1);
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
	T* raw();
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
/// A = b * b
/// by relatively robust representations
/// A is symmetric and positive semi-definite matrix
/// A = Z⋅D⋅ZT
/// D = D'⋅D'
/// b = Z⋅D'⋅ZT
	void sqrtRRR(DenseMatrix<T>& b) const;
	
	friend std::ostream& operator<<(std::ostream &output, const DenseMatrix<T> & p) {
        output << p.str();
        return output;
    }
	
	static void PrintMatrix(char* desc, int m, int n, T* a)
	{
		int i, j;
		std::cout<<"\n "<<desc;
		for(i=0; i< m; i++) {
			std::cout<<"\n| ";
			for(j=0; j< n; j++) {
				std::cout<<" "<<a[j*m + i];
			}
			std::cout<<" |";
		}
		std::cout<<"\n";
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
T* DenseMatrix<T>::raw()
{ return m_v; }

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
	clapack_syrk<T>("U", "T", m_numColumns, m_numRows, 
										T(1.0), m_v, m_numRows, 
										T(0.0), dst.m_v, m_numColumns);
    dst.fillSymmetric();
}

template <typename T>
void DenseMatrix<T>::mult(DenseVector<T>& b, const DenseVector<T>& x, 
            const T alpha, const T beta) const
{
	clapack_gemv<T>("N", m_numRows, m_numColumns, 
							alpha, m_v, m_numRows, 
							x.v(), 1, 
							beta, b.v(), 1);
}

template <typename T>
void DenseMatrix<T>::multTrans(DenseVector<T>& b, const DenseVector<T>& x, 
            const T alpha, const T beta) const
{
	clapack_gemv<T>("T", m_numRows, m_numColumns, 
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
	const int n = min(m_numRows, m_numColumns);
	for(int i = 0; i<n; ++i) 
		m_v[i*m_numRows+i] += diag; 
};

/// http://scc.qibebt.cas.cn/docs/library/Intel%20MKL/2011/mkl_manual/lse/functn_syevr.htm
/// all eigenvalues and eigenvectors
template <typename T> 
void DenseMatrix<T>::sqrtRRR(DenseMatrix<T>& b) const
{
	T * W = new T[m_numRows];
	T * Z = new T[m_numRows*m_numRows];
	integer * ISuppz = new integer[2*m_numRows];
	
	T * work;
	integer * iwork;
	T abstol = -1.0;
	T vl, vu;
	int il = 1;
	int iu = m_numRows;
	integer m;
	integer info;
	integer lwork = -1;
	integer liwork = -1;
	T queryWork; work = &queryWork;
	integer queryIwork; iwork = &queryIwork;
	
	clapack_syevr<T>("V", "A", "U", m_numRows, m_v, m_numRows, 
		&vl, &vu, il, iu, abstol, &m,
         W, Z, m_numRows, ISuppz, 
		 work, &lwork, iwork, &liwork, &info);
		 
	lwork = queryWork;
	liwork = queryIwork;
	
	work = new T[lwork];
	iwork = new integer[liwork];
	
	clapack_syevr<T>("V", "A", "U", m_numRows, m_v, m_numRows, 
		&vl, &vu, il, iu, abstol, &m,
         W, Z, m_numRows, ISuppz, 
		 work, &lwork, iwork, &liwork, &info);
		 
/// B = Z * D		 
	T * B = new T[m_numRows*m_numRows];
	int i, j;
    for(i=0; i< m_numRows; i++) {
		double  lambda=sqrt(W[i]);
        for(j=0; j< m_numRows; j++) {
            B[i*m_numRows + j] = Z[i*m_numRows + j] * lambda;
        }
    }

/// b = B * ZT	
	clapack_gemm<double>("N", "T", m_numRows, m_numRows, m_numRows,
						1.0, B, m_numRows, Z, m_numRows, 0.0, b.raw(), m_numRows);
						
	delete[] work;
	delete[] iwork;
	delete[] B;
	delete[] W;
	delete[] Z;
	delete[] ISuppz;
}

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


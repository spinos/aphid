#ifndef APHID_TYPES_H
#define APHID_TYPES_H

#include <iostream>
#include <sstream>
#include <string.h>

namespace aphid {

struct Color4 {
	Color4(): r(0.f), g(0.f), b(0.f), a(0.f)
	{}
	
	Color4(float x, float y, float z, float w)
	: r(x), g(y), b(z), a(w)
	{}
	
	float r, g, b, a;
};

struct Float4 {
	Float4(): x(0.f), y(0.f), z(0.f), w(0.f)
	{}
	
	Float4(float a, float b, float c, float d)
	: x(a), y(b), z(c), w(d)
	{}
	
	void set(float a, float b, float c, float d)
	{ x = a; y = b; z = c; w = d; }
	
	Float4 operator * ( const float& scale ) const
	{
		return Float4(
		x * scale,
		y * scale,
		z * scale,
		w * scale);
	}
	
	Float4 operator + ( const Float4 & b ) const
	{
		return Float4(
		x + b.x,
		y + b.y,
		z + b.z,
		w + b.w);
	}
	
	float dot(const Float4 & b) const
	{
		return x * b.x + y * b.y + z * b.z + w * b.w;
	}
	
	float x, y, z, w;
};

struct Float3 {
	Float3(): x(0.f), y(0.f), z(0.f)
	{}
	
	Float3(float a, float b, float c)
	: x(a), y(b), z(c)
	{}
	
	void set(float a, float b, float c) { x = a; y = b; z = c; }
	
	float x, y, z;
};

struct Float2 {
	Float2(): x(0.f), y(0.f)
	{}
	
	Float2(float a, float b)
	: x(a), y(b)
	{}
	
	void set(float a, float b) { x = a; y = b; }
	
	float x, y;
};

struct Int2 {
	
	Int2(): x(0), y(0)
	{}
	
	Int2(int a, int b)
	: x(a), y(b)
	{}
	
	void set(int a, int b) 
	{ x = a; y = b; }
	
	int x, y;
};

template<typename T>
struct VectorN {
	VectorN() {
		_ndim = 0;
		_data = 0;
	}
	
	VectorN(const int & n) {
		_ndim = n;
		_data = new T[n];
	}
	
	~VectorN() {
		if(_data) delete[] _data;
	}
	
	const int & N() const
	{ return _ndim; }
	
	const T * v() const {
		return _data;
	}
	
	T * v() {
		return _data;
	}
	
	void create(const int & n)
	{
		if(_ndim < n) {
			if(_ndim > 0)
				delete[] _data;
			_data = new T[n];
		}
		_ndim = n;
	}
	
	void create(const T * x, const int & n)
	{
		create(n);
		memcpy(_data, x, sizeof(T) * n);
	}
	
	int periodic(const int & i) const {
		if(i<0)
			return _ndim +i;
		if(i>_ndim-1)
			return i-_ndim;
		return i;
	}
	
	void copy(const T * v, const int & n, const int & p=0) {
		create(n);
		if(p==0) {
			memcpy(_data, v, sizeof(T) * n);
			return;
		}
		
		const int q = p>0 ? p : -p;

		if(p<0) {
			memcpy(_data, &v[q], sizeof(T) * (n-q) );
			memcpy(&_data[n-q], v, sizeof(T) * q );
		}
		else {
			memcpy(&_data[q], v, sizeof(T) * (n-q) );
			memcpy(_data, &v[n-q], sizeof(T) * q );
		}
	}
	
/// copy with shift phase
	void copy(const VectorN<T> & another, const int & p = 0) {
		copy(another.v(), another.N(), p);
	}
	
	void setZero(const int & n) {
		create(n);
		memset(_data, 0, sizeof(T) * _ndim);
	}
	
	void operator=(const VectorN<T> & another) {
		copy(another);
	}
	
	const T & operator[](const int & i) const {
		return _data[i];
	}
	
	T & operator[](const int & i) {
		return _data[i];
	}
	
	T* at(const int & i) const {
		return &_data[i];
	}
	
	VectorN<T> operator+(const VectorN<T> & another) const {
		VectorN<T> r(_ndim);
		for(int i = 0; i < _ndim; i++) *r.at(i) = _data[i] + another[i];
		return r;
	}
	
	VectorN<T> operator-(const VectorN<T> & another) const {
		VectorN<T> r(_ndim);
		for(int i = 0; i < _ndim; i++) *r.at(i) = _data[i] - another[i];
		return r;
	}
	
	VectorN<T> operator*(const T & scale) const {
		VectorN<T> r(_ndim);
		for(int i = 0; i < _ndim; i++) *r.at(i) = _data[i] * scale;
		return r;
	}
	
	void operator*=(const T & scale) {
		for(int i = 0; i < _ndim; i++) 
			_data[i] *= scale;
	}
	
	void operator+=(const VectorN<T> & another) {
		for(int i = 0; i < _ndim; i++) 
			_data[i] += another[i];
	}
	
	T multiplyTranspose() const {
		T r;
		for(int i = 0; i < _ndim; i++) r += _data[i] * _data[i];
		return r;
	}
	
	void maxAbsError(T & err, const VectorN & another) const {
		int i = 0;
		for(;i<_ndim;++i) {
			T d = _data[i] - another.v()[i];
			if(d < 0) d = -d;
			if(err < d)
				err = d;
		}
	}
	
/// decrease sampling rate by integer p with phase offset
/// X[N] input signal
	void downsample(const T * x, const int & n, 
					const int & p, const int & phase=0) {
	
		int i=0, j=0;
		for(;i<n;++i) {
			if(i== j*p + phase) {
				j++;
			}
		}
		
		create(j);
		
		i=j=0;
		for(;i<n;++i) {
			if(i== j*p + phase) {
				v()[j++]=x[i];
			}
		}
	}
	
/// P phase of shift |P| < N
/// delay the signal when P > 0	
	void circshift(const int & p) {
		if(p==0) 
			return;
		
		VectorN<float> b;
		b.copy(*this);
		copy(b, p);
		
	}
	
	std::string info() const {
		std::stringstream sst;
		sst.str("");
		sst<<"(";
		for(int i = 0; i < _ndim - 1; i++) sst<<_data[i]<<", ";
		sst<<_data[_ndim - 1]<<")";
		return sst.str();
	}
	
	int _ndim;
	T * _data;
};

/// transform and out-of-bound return value
template<typename T>
struct Array2SampleProfile {

	Float2 _translate;
	Float2 _scale;
	T _defaultValue;
	
};

/// m-by-n array
/// column major
/// 0   m   ... (n-1)m
/// 1   m+1     (n-1)m+1
/// .   .       .
/// .   .       .
/// .   .       .
/// m-1 2m-1... nm-1
/// m number of rows
/// n number of columns
template<typename T>
struct Array2 {

	T * m_data;
	int m_M, m_N;
	
	Array2() {
		m_M = m_N = 0;
		m_data = NULL;
	}
	
	Array2(const Array2<T> & another) {
		m_M = m_N = 0;
		m_data = NULL;
		copy(another);
	}
	
	~Array2() {
		if(m_M>0) delete[] m_data;
	}
	
	void create(const int & m, const int & n) {
		if(m_M * m_N < m*n) {
			if(m_M) delete[] m_data;
			m_data = new T[m*n];
		}
		
		m_M = m;
		m_N = n;
	}
	
	const int & numRows() const {
		return m_M;
	}
	
	const int & numCols() const {
		return m_N;
	}
	
	const T * v() const {
		return m_data;
	}
	
	T * v() {
		return m_data;
	}

	const T * column(const int & i) const {
		return &m_data[i*m_M];
	}
	
	T * column(const int & i) {
		return &m_data[i*m_M];
	}
	
/// u column v row
	int iuv(const int & u, const int & v) const {
		return u * m_M + v;
	}
	
	void operator=(const Array2<T> & another) {
		copy(another);
	}
	
	void copy(const Array2<T> & another) {
		create(another.numRows(), another.numCols() );
		memcpy(m_data, another.v(), m_M*m_N*sizeof(T) );
	}
	
	void setZero() {
		memset (m_data, 0, m_M*m_N*sizeof(T) );
	}
	
	void copyColumn(const int & i, const T * b) {
		memcpy(column(i), b, m_M *sizeof(T) );
	}
	
	void transpose() {
		Array2 old(*this);
		
		int s = m_M;
		m_M = m_N;
		m_N = s;
		
		const T * src = old.v();
		
		int i, j;
		for(j = 0;j<m_N;++j) {
			
			T * dst = column(j);
			for(i=0;i<m_M;++i) {
				dst[i] = src[i * m_N + j];
			}
		}
	}
	
	void maxAbsError(T & err, const Array2 & another) const {
		const int mn = m_M * m_N;
		int i = 0;
		for(;i<mn;++i) {
			T d = m_data[i] - another.v()[i];
			if(d < 0) d = -d;
			if(err < d)
				err = d;
		}
	}
	
	int maxDim() const {
		return m_M > m_N ? m_M : m_N;
	}
	
	Array2<T> operator+(const Array2<T> & another) const {
		Array2<T> r;
		r.copy(*this);
		r += another;
		return r;
	}
	
	Array2<T> operator-(const Array2<T> & another) const {
		Array2<T> r;
		r.copy(*this);
		r -= another;
		return r;
	}
	
	void operator+=(const Array2<T> & another) {
		const int mn = m_M * m_N;
		for(int i = 0; i < mn;++i) 
			m_data[i] += another.v()[i];
	}
	
	void operator-=(const Array2<T> & another) {
		const int mn = m_M * m_N;
		for(int i = 0; i < mn;++i) 
			m_data[i] -= another.v()[i];
	}
	
	void operator*=(const T & s) {
		const int mn = m_M * m_N;
		for(int i = 0; i < mn; i++) 
			m_data[i] *= s;
	}
	
	friend std::ostream& operator<<(std::ostream &output, const Array2 & p) {
        output << p.str();
        return output;
    }
	
	const std::string str() const {
		std::stringstream sst;
		sst.str("");
		
		for(int j=0;j<m_M;++j) {
			sst<<"\n|";
			for(int i=0;i<m_N;++i) {
				sst<<m_data[iuv(i,j)];
				if(i<m_N-1)
					sst<<", ";
			}
			sst<<"|";
		}
		return sst.str();
	}
	
/// get a part of 
	void sub(Array2 & d,
			const int & m0, const int & n0) const {
		
		const int & ms = d.numRows();
		const int & ns = d.numCols();
		
		for(int i=0;i<ns;++i) {
			memcpy(d.column(i), &column(n0+i)[m0], sizeof(T) * ms);
		}
	}
	
	void sample(const Array2<T> & another, 
		const Array2SampleProfile<T> * prof);
	
/// map value by coord u in row v in column
/// return false if coord is out-of-bound
	bool getValue(T & dst, const float & u, const float & v) const;
	
	void convoluteVertical(const Array2<T> & src,
                const T * kern3tap);
	void convoluteHorizontal(const Array2<T> & src,
                const T * kern3tap);
/// vmin vmax already have values				
	void getMinMax(T & vmin, T & vmax) const;
/// set all values
	void set(const T & x);
	
};

template<typename T>
bool Array2<T>::getValue(T & dst, const float & u, const float & v) const
{
	if(u < 0.f || u > 1.f) {
		return false;
	}
	if(v < 0.f || v > 1.f) {
		return false;
	}
	
	float fu = u * (float)m_N - .5f;
	float fv = v * (float)m_M - .5f;
	int u0 = fu;
	int v0 = fv;
	int u1 = u0 + 1;
	int v1 = v0 + 1;
	
	if(u0 < 0) {
		u0 = 0;
	}
	if(v0 < 0) {
		v0 = 0;
	}
	if(u1 > m_N - 1) {
		u1 = m_N - 1;
	}
	if(v1 > m_M - 1) {
		v1 = m_M - 1;
	}
	
	fu -= u0;
	fv -= v0;
	if(fu < 0.f) {
		fu = 0.f;
	}
	if(fv < 0.f) {
		fv = 0.f;
	}
	
	T box[4];
	box[0] = m_data[u0 * m_M + v0];
	box[1] = m_data[u0 * m_M + v1];
	box[2] = m_data[u1 * m_M + v0];
	box[3] = m_data[u1 * m_M + v1];
	
	box[0] += fv * (box[1] - box[0]);
	box[2] += fv * (box[3] - box[2]);
		
	dst = box[0] + fu * (box[2] - box[0]);
	
	return true;
}

template<typename T>
void Array2<T>::sample(const Array2<T> & another, 
		const Array2SampleProfile<T> * prof)
{
	const float di = 1.f / (float)(m_M);
	const float dj = 1.f / (float)(m_N);
	const float hdi = di * .5f;
	const float hdj = dj * .5f;
	
	float u, v;
	
	for(int j=0;j<m_N;++j) {
		u = (dj * j + hdj) * prof->_scale.x + prof->_translate.x;
		
		T * dc = column(j);
		for(int i=0;i<m_M;++i) {
		
			v = (di * i + hdi) * prof->_scale.y + prof->_translate.y;
			
			if(!another.getValue(dc[i], u, v) ) {
				dc[i] = prof->_defaultValue;
			}
		}
	}
}

template<typename T>
void Array2<T>::convoluteVertical(const Array2<T> & src,
                const T * kern3tap)
{
	const int & m = numRows();
	const int & n = numCols();
	const int limi = m - 1;
	int ri;
	for(int j=0;j<n;++j) {
		
		const T * srcColj = src.column(j);
		T * colj = column(j);
		
		for(int i=0;i<m;++i) {
		
			T & vi = colj[i];
			
			vi = (T)0.0;
			
			ri = i - 1;
			if(ri < 0) {
				ri = 0;
			}
			
			vi += srcColj[ri] * kern3tap[0];
			
			vi += srcColj[i] * kern3tap[1];
			
			ri = i + 1;
			if(ri > limi) {
				ri = limi;
			}
			
			vi += srcColj[ri] * kern3tap[2];
			
		}
	}
}

template<typename T>
void Array2<T>::convoluteHorizontal(const Array2<T> & src,
                const T * kern3tap)
{
	const int & m = numRows();
	const int & n = numCols();
	const int limj = n - 1;
	int j0, j1;
	for(int j=0;j<n;++j) {
		
		T * colj = column(j);
		
		const T * srcColj = src.column(j);
		
		j0 = j - 1;
		if(j0 < 0) {
			j0 = 0;
		}
		
		const T * srcColj0 = src.column(j0);
		
		j1 = j + 1;
		if(j1 > limj) {
			j1 = limj;
		}
		
		const T * srcColj1 = src.column(j1);
		
		for(int i=0;i<m;++i) {
		
			T & vi = colj[i];
			
			vi = (T)0.0;
			
			vi += srcColj0[i] * kern3tap[0];
			vi += srcColj[i] * kern3tap[1];
			vi += srcColj1[i] * kern3tap[2];
			
		}
	}
	
}

template<typename T>
void Array2<T>::getMinMax(T & vmin, T & vmax) const
{
	const int mn = numCols() * numRows();
	for(int i=0;i<mn;++i) {
		const T & vi = m_data[i];
		if(vmin > vi) {
			vmin = vi;
		}
		if(vmax < vi) {
			vmax = vi;
		}
	}
}

template<typename T>
void Array2<T>::set(const T & x)
{
	const int mn = numCols() * numRows();
	for(int i=0;i<mn;++i) {
		m_data[i] = x;
	}
}

/// http://www.owlnet.rice.edu/~ceng303/manuals/fortran/FOR5_3.html
/// m-by-n-by-p array
template<typename T>
struct Array3 {
	
	Array2<T> * m_slice;
	int m_P;
	
	Array3() {
		m_slice = NULL;
		m_P = 0;
	}
	
	Array3(const Array3 & another) {
		m_slice = NULL;
		m_P = 0;
		copy(another);
	}
	
	~Array3() {
		if(m_P) delete[] m_slice;
	}
	
	void create(const int & m, const int & n, const int & p = 1) {
		if(m_P < p) {
			if(m_P) delete[] m_slice;
			m_slice = new Array2<T>[p];	
		}
		
		for(int i=0; i<p; ++i)
			m_slice[i].create(m,n);
		
		m_P = p;
	}

	const int & numRows() const {
		return m_slice[0].numRows();
	}
	
	const int & numCols() const {
		return m_slice[0].numCols();
	}
	
	const int & numRanks() const {
		return m_P;
	}
	
/// i-th rank
	const Array2<T> * rank(const int & i) const {
		return &m_slice[i];
	}

	Array2<T> * rank(const int & i) {
		return &m_slice[i];
	}
	
	void copy(const Array3<T> & another) {
		create(another.numRows(), another.numCols(),
				another.numRanks() );
		for(int i=0;i<another.numRanks();++i) {
			rank(i)->copy(*another.rank(i) );
		}
	}
	
	void operator=(const Array3<T> & another) {
		copy(another);
	}
	
	void setZero();
	
	void sample(const Array3<T> & another, 
		const Array2SampleProfile<T> * prof);
};

template<typename T>
void Array3<T>::setZero()
{
	for(int i=0;i<m_P;++i) {
		m_slice[i].setZero();
	}
}

template<typename T>
void Array3<T>::sample(const Array3<T> & another, 
		const Array2SampleProfile<T> * prof)
{
	for(int i=0;i<m_P;++i) {
		m_slice[i].sample(*another.rank(i), prof);
	}
}

}
#endif        //  #ifndef APHID_TYPES_H

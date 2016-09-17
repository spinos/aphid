#ifndef APHID_TYPES_H
#define APHID_TYPES_H
#include <sstream>

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
	
/// copy with shift phase
	void copy(const VectorN<T> & another, const int & p = 0) {
		const int & n = another.N();
		create(n);
		
		if(p==0) {
			memcpy(_data, another.v(), sizeof(T) * n);
			return;
		}
		
		const int q = p>0 ? p : -p;

		if(p<0) {
			memcpy(_data, &another.v()[q], sizeof(T) * (n-q) );
			memcpy(&_data[n-q], another.v(), sizeof(T) * q );
		}
		else {
			memcpy(&_data[q], another.v(), sizeof(T) * (n-q) );
			memcpy(_data, &another.v()[n-q], sizeof(T) * q );
		}
		
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
	
};

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
};

}
#endif        //  #ifndef APHID_TYPES_H
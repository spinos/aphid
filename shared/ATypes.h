#ifndef ATYPES_H
#define ATYPES_H
#include <sstream>
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

template<typename T>
struct VectorN {
	VectorN() {
		_ndim = 0;
		_data = 0;
	}
	
	VectorN(unsigned n) {
		_ndim = n;
		_data = new T[n];
	}
	
	~VectorN() {
		if(_data) delete[] _data;
	}
	
	void setZero(unsigned n) {
		if(_data) delete[] _data;
		_ndim = n;
		_data = new T[n];
		for(unsigned i = 0; i < _ndim; i++) _data[i] = 0;
	}
	
	void operator=(const VectorN<T> & another) {
		if(_data) delete[] _data;
		_ndim = another._ndim;
		_data = new T[_ndim];
		for(unsigned i = 0; i < _ndim; i++) _data[i] = another[i];
	}
	
	T operator[](unsigned i) const {
		return _data[i];
	}
	
	T* at(unsigned i) const {
		return &_data[i];
	}
	
	VectorN<T> operator+(const VectorN<T> & another) const {
		VectorN<T> r(_ndim);
		for(unsigned i = 0; i < _ndim; i++) *r.at(i) = _data[i] + another[i];
		return r;
	}
	
	VectorN<T> operator-(const VectorN<T> & another) const {
		VectorN<T> r(_ndim);
		for(unsigned i = 0; i < _ndim; i++) *r.at(i) = _data[i] - another[i];
		return r;
	}
	
	VectorN<T> operator*(const T & scale) const {
		VectorN<T> r(_ndim);
		for(unsigned i = 0; i < _ndim; i++) *r.at(i) = _data[i] * scale;
		return r;
	}
	
	T multiplyTranspose() const {
		T r;
		for(unsigned i = 0; i < _ndim; i++) r += _data[i] * _data[i];
		return r;
	}
	
	std::string info() const {
		std::stringstream sst;
		sst.str("");
		sst<<"(";
		for(unsigned i = 0; i < _ndim - 1; i++) sst<<_data[i]<<", ";
		sst<<_data[_ndim - 1]<<")";
		return sst.str();
	}
	
	unsigned _ndim;
	T * _data;
};
#endif        //  #ifndef ATYPES_H


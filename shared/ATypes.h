#ifndef ATYPES_H
#define ATYPES_H

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
	
	float x, y, z, w;
};

struct Float3 {
	Float3(): x(0.f), y(0.f), z(0.f)
	{}
	
	Float3(float a, float b, float c)
	: x(a), y(b), z(c)
	{}
	
	float x, y, z;
};
#endif        //  #ifndef ATYPES_H


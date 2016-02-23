#ifndef BASEBUFFER_H
#define BASEBUFFER_H

/*
 *  BaseBuffer.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
namespace aphid {

class BaseBuffer {
public:	
	BaseBuffer();
	virtual ~BaseBuffer();
	
	void create(unsigned size);
	void destroy();
	
	const unsigned bufferName() const;
	const unsigned bufferSize() const;
	char * data() const;
    void copyFrom(const void * src);
    void copyFrom(const void * src, unsigned size);
	void copyFrom(const void * src, unsigned size, unsigned loc);
    
    template<typename T>
    void minus(const BaseBuffer * other, unsigned n)
    {
        T * dst = (T *)data();
        T * src = (T *)other->data();
        unsigned i = 0;
        for(;i<n;i++) dst[i] -= src[i];
    }
protected:

private:
    char * m_native;
	unsigned m_bufferSize;
};

class TypedBuffer : public BaseBuffer {
public:
	enum ValueType {
		TUnknown = 0,
		TChar = 1,
		TShort = 2,
		TFlt = 4,
		TVec2 = 8,
		TVec3 = 12,
		TVec4 = 16
	};
	
	TypedBuffer();
	virtual ~TypedBuffer();
	
	void create(ValueType t, unsigned size);
	
	ValueType valueType() const;
    
    template<typename T>
    T * typedData() const
    { return (T *)data(); }
	
	unsigned numElements() const;
	
    template<typename T, typename Ts> 
	T sample(Ts * s) const 
	{ return s->evaluate(typedData<T>()); }
    
    void operator-=( const BaseBuffer * other );

protected:
    void minusFlt(const BaseBuffer * other);
    void minusVec3(const BaseBuffer * other);
private:
	ValueType m_type;
};

}
#endif        //  #ifndef BASEBUFFER_H

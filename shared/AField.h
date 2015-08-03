/*
 *  AField.h
 *  larix
 *
 *  Created by jian zhang on 8/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <AllMath.h>
#include <map>
#include <string>
#include "BaseBuffer.h"

class AField {
public:
	template<typename T>
	class BaseSampler {
	public:
		BaseSampler() {}
		
		virtual T evaluate(T * data) const
		{ return T(); }
	};
	
	template<typename T>
	class LineSampler : public BaseSampler<T> {
	public:
		LineSampler() {}
		
		unsigned * vertices() 
		{ return m_vertices; }
		
		float * contributes() 
		{ return m_contributes; }
		
		virtual T evaluate(T * data) const 
		{
			return data()[m_vertices[0]] * m_contributes[0]
					+ data()[m_vertices[1]] * m_contributes[1];
		}
		
	private:
		unsigned m_vertices[2];
		float m_contributes[2];
	};
	
	template<typename T>
	class TriangleSampler : public BaseSampler<T> {
	public:
		TriangleSampler() {}
		
		unsigned * vertices() 
		{ return m_vertices; }
		
		float * contributes() 
		{ return m_contributes; }
		
		virtual T evaluate(T * data) const 
		{
			return data()[m_vertices[0]] * m_contributes[0]
					+ data()[m_vertices[1]] * m_contributes[1]
					+ data()[m_vertices[2]] * m_contributes[2];
		}
		
	private:
		unsigned m_vertices[3];
		float m_contributes[3];
	};
	
	template<typename T>
	class TetrahedronSampler : public BaseSampler<T> {
	public:
		TetrahedronSampler() {}
		
		unsigned * vertices() 
		{ return m_vertices; }
		
		float * contributes() 
		{ return m_contributes; }
		
		virtual T evaluate(T * data) const 
		{
			return data()[m_vertices[0]] * m_contributes[0]
					+ data()[m_vertices[1]] * m_contributes[1]
					+ data()[m_vertices[2]] * m_contributes[2]
					+ data()[m_vertices[3]] * m_contributes[3];
		}
		
	private:
		unsigned m_vertices[4];
		float m_contributes[4];
	};
	
	AField();
	virtual ~AField();
	
	void addFloatChannel(const std::string & name, unsigned n);
	void addVec3Channel(const std::string & name, unsigned n);
	
	bool useChannel(const std::string & name);
	
	float * fltValue() const;
	Vector3F * vec3Value() const;
	
	template<typename T> 
	T sample(BaseSampler<T> * sampler) const 
	{ return sampler->evaluate(value()); }
	
protected:
	template<typename T> 
	T * value() const
	{ return (T *)m_currentChannel->data(); }

private:
	std::map<std::string, BaseBuffer * > m_channels;
	BaseBuffer * m_currentChannel;
};
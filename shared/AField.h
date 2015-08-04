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
#include "BaseSampler.h"
class AField {
public:	
	AField();
	virtual ~AField();
	
	void addFloatChannel(const std::string & name, unsigned n);
	void addVec3Channel(const std::string & name, unsigned n);
	
	bool useChannel(const std::string & name);
	
	float * fltValue() const;
	Vector3F * vec3Value() const;
	
	template<typename T, typename Ts> 
	T sample(Ts * s) const 
	{ return s->evaluate<T>(currentValue<T>()); }
	
	void getChannelNames(std::vector<std::string > & names) const;
	TypedBuffer * currentChannel() const;
	TypedBuffer * namedChannel(const std::string & name);
	unsigned numChannels() const;
protected:
	template<typename T> 
	T * currentValue() const
	{ return (T *)m_currentChannel->data(); }

private:
	std::map<std::string, TypedBuffer * > m_channels;
	TypedBuffer * m_currentChannel;
};
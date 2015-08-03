/*
 *  AField.cpp
 *  larix
 *
 *  Created by jian zhang on 8/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AField.h"

AField::AField() {}
AField::~AField() 
{
	std::map<std::string, TypedBuffer * >::iterator it = m_channels.begin();
	for(;it!=m_channels.end();++it) delete it->second;
	m_channels.clear();
}

void AField::addFloatChannel(const std::string & name, unsigned n)
{
	TypedBuffer * b = new TypedBuffer;
	b->create(TypedBuffer::TFlt, n*4);
	m_channels[name] = b;
}

void AField::addVec3Channel(const std::string & name, unsigned n)
{
	TypedBuffer * b = new TypedBuffer;
	b->create(TypedBuffer::TVec3, n*12);
	m_channels[name] = b;
}

bool AField::useChannel(const std::string & name)
{
	std::map<std::string, TypedBuffer * >::iterator it = m_channels.find(name);
	if(m_channels.find(name) == m_channels.end())
		return false;
	
	m_currentChannel = it->second;
	return true;
}

float * AField::fltValue() const
{ return value<float>(); }

Vector3F * AField::vec3Value() const
{ return value<Vector3F>(); }

void AField::getChannelNames(std::vector<std::string > & names) const
{
	std::map<std::string, TypedBuffer * >::const_iterator it = m_channels.begin();
	for(;it!=m_channels.end();++it) names.push_back(it->first);
}

TypedBuffer * AField::currentChannel() const
{ return m_currentChannel; }

TypedBuffer * AField::namedChannel(const std::string & name)
{ return m_channels[name]; }
//:~
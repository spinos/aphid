/*
 *  AField.cpp
 *  larix
 *
 *  Created by jian zhang on 8/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AField.h"

namespace aphid {

AField::AField() {}
AField::~AField() 
{
	std::map<std::string, TypedBuffer * >::iterator it = m_channels.begin();
	for(;it!=m_channels.end();++it) delete it->second;
	m_channels.clear();
}

AField::FieldType AField::fieldType() const
{ return FldBase; }

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
{ return currentValue<float>(); }

Vector3F * AField::vec3Value() const
{ return currentValue<Vector3F>(); }

void AField::getChannelNames(std::vector<std::string > & names) const
{
	std::map<std::string, TypedBuffer * >::const_iterator it = m_channels.begin();
	for(;it!=m_channels.end();++it) names.push_back(it->first);
}

TypedBuffer * AField::currentChannel() const
{ return m_currentChannel; }

TypedBuffer * AField::namedChannel(const std::string & name)
{ 
    if(m_channels.find(name) == m_channels.end()) return 0;
    return m_channels[name]; 
}

char * AField::namedData(const std::string & name)
{
    TypedBuffer * chan = namedChannel(name);
    if(!chan) return 0;
    return chan->data();
}

unsigned AField::numChannels() const
{ return m_channels.size(); }

void AField::setChannelZero(const std::string & name)
{
    TypedBuffer * chan = namedChannel(name);
    
    if(chan->valueType() == TypedBuffer::TFlt)
		setFltChannelZero(chan);
	else if(chan->valueType() == TypedBuffer::TVec3)
		setVec3ChannelZero(chan);
}

void AField::setFltChannelZero(TypedBuffer * chan)
{
    float * dst = (float *)chan->data();
    const unsigned n = chan->bufferSize() / 4;
    unsigned i=0;
    for(;i<n;i++) dst[i] = 0.f;
}

void AField::setVec3ChannelZero(TypedBuffer * chan)
{
    Vector3F * dst = (Vector3F *)chan->data();
    const unsigned n = chan->bufferSize() / 12;
    unsigned i=0;
    for(;i<n;i++) dst[i].setZero();
}

void AField::verbose() const
{
    std::cout<<"\n field:"
    <<"\n n channels "<<numChannels();
    
    std::map<std::string, TypedBuffer * >::const_iterator it = m_channels.begin();
	for(;it!=m_channels.end();++it)
         std::cout<<"\n channel[\""<<it->first<<"\"]";
}

}
//:~
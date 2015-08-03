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
	std::map<std::string, BaseBuffer * >::iterator it = m_channels.begin();
	for(;it!=m_channels.end();++it) delete it->second;
	m_channels.clear();
}

void AField::addFloatChannel(const std::string & name, unsigned n)
{
	BaseBuffer * b = new BaseBuffer;
	b->create(n*4);
	m_channels[name] = b;
}

void AField::addVec3Channel(const std::string & name, unsigned n)
{
	BaseBuffer * b = new BaseBuffer;
	b->create(n*12);
	m_channels[name] = b;
}

bool AField::useChannel(const std::string & name)
{
	std::map<std::string, BaseBuffer * >::iterator it = m_channels.find(name);
	if(m_channels.find(name) == m_channels.end())
		return false;
	
	m_currentChannel = it->second;
	return true;
}

float * AField::fltValue() const
{ return value<float>(); }

Vector3F * AField::vec3Value() const
{ return value<Vector3F>(); }
//:~
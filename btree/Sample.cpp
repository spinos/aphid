/*
 *  Sample.cpp
 *  btree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Sample.h"

namespace sdb {

ChannelArray::ChannelArray() 
{
	m_dataSize = 0;
}

void ChannelArray::addChannel(const std::string & name, ChannelType type)
{
	if(m_channelStarts.find(name) != m_channelStarts.end()) return;
	ChannelRange r;
	r.begin = m_dataSize;
	r.length = type;
	m_channelStarts[name] = r;
	m_dataSize += type;
}

void ChannelArray::chooseChannel(const std::string & name)
{
	m_activeChannel = m_channelStarts[name];
}

int ChannelArray::dataSize() const { return m_dataSize; }

ChannelArray * Sample::SampleChannels = NULL;

Sample::~Sample() 
{
	if(m_data) delete[] m_data;
}

void Sample::setChannels(ChannelArray * channels)
{
	SampleChannels = channels;
	int s = channels->dataSize();
	if(s % 32 > 0) s = 32 * (s/32 + 1);
	m_data = new char[s];
}

void Sample::set(float * x)
{

}

} // end namespace sdb
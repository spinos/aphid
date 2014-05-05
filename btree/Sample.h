/*
 *  Sample.h
 *  btree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <string>
#include <map>
#include <boost/format.hpp>
namespace sdb {

class ChannelArray {
public:
	enum ChannelType {
		TFloat = 4,
		TV2 = 8,
		TV3 = 12
	};
	
	struct ChannelRange {
		int begin, length;
	};
	
	ChannelArray();
	
	void addChannel(const std::string & name, ChannelType type);
	void chooseChannel(const std::string & name);
	int dataSize() const;
private:
	std::map<std::string, ChannelRange> m_channelStarts;
	ChannelRange m_activeChannel;
	int m_dataSize;
};

class Sample {
public:
	Sample() : m_data(NULL) {}
	virtual ~Sample();

	void setChannels(ChannelArray * channels);
	void set(float * x);
private:	
	static ChannelArray * SampleChannels;
	char * m_data;
};

} // end namespace sdb
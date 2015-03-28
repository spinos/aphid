/*
 *  BaseLog.cpp
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/29/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseLog.h"
#include "boost/date_time/posix_time/posix_time.hpp"
using namespace boost::posix_time;
BaseLog::BaseLog(const std::string & fileName)
{
	m_file.open(fileName.c_str(), std::ios::app);
	m_file<<"start logging ";
	writeTime();
}

BaseLog::~BaseLog() 
{
	if(m_file.is_open())
		m_file.close();
}

void BaseLog::write(const std::string & os )
{ m_file<<os; }

void BaseLog::writeTime()
{ 
	const ptime now = second_clock::local_time();
	m_file<<" at "<<to_iso_extended_string(now)<<"\n";
}
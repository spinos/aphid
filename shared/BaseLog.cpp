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
#include <boost/format.hpp>

using namespace boost::posix_time;
BaseLog::BaseLog(const std::string & fileName)
{
	m_file.open(fileName.c_str(), std::ios::trunc);
	m_file<<"start logging ";
	writeTime();
}

BaseLog::~BaseLog() 
{
	if(m_file.is_open())
		m_file.close();
}

void BaseLog::write(unsigned & i)
{ _write<unsigned>(i); }

void BaseLog::write(const std::string & os )
{ m_file<<os; }

void BaseLog::writeTime()
{ 
	const ptime now = second_clock::local_time();
	m_file<<" at "<<to_iso_extended_string(now)<<"\n";
}

void BaseLog::newLine()
{ m_file<<"\n"; }

void BaseLog::writeArraySize(const unsigned & n)
{ write(boost::str(boost::format(" [%1%] \n") % n)); }

void BaseLog::writeStruct1(char * data, const std::vector<std::pair<int, int> > & desc)
{
    std::vector<std::pair<int, int> >::const_iterator it = desc.begin();
    for(;it!=desc.end();++it)
        writeByTypeAndLoc(it->first, it->second, data);
    newLine();
}

void BaseLog::writeByTypeAndLoc(int type, int loc, char * data)
{
    switch(type) {
    case 0:
        _write<int>(*(int *)&data[loc]);
        break;
    case 1:
        _write<float>(*(float *)&data[loc]);
        break;
    default:
        ;
    }
}


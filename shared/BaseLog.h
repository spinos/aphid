#ifndef BASELOG_H
#define BASELOG_H

/*
 *  BaseLog.h
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/29/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <map>
#include <boost/format.hpp>

class BaseLog {
public:
	enum Frequency {
		FIgnore = 0,
        FOnce = 1,
        FAlways = 2
    };
	BaseLog(const std::string & fileName);
	virtual ~BaseLog();
	
	void write(const std::string & os);
	void write(unsigned & i);
	void writeTime();
	void newLine();
	void writeArraySize(const unsigned & n);
	void writeArrayIndex(const unsigned & n);
	void writeStruct1(char * data, const std::vector<std::pair<int, int> > & desc);
	void braceBegin(const std::string & notation, Frequency freq = FOnce);
	void braceEnd(const std::string & notation, Frequency freq = FOnce);
	template <typename T>
	static std::string addSuffix(const std::string & name, const T & a)
	{
		return boost::str(boost::format("%1%_%2%\n") % name % a);
	}
	template <typename T>
	static std::string addPrefix(const std::string & name, const T & a)
	{
		return boost::str(boost::format("%1%_%2%\n") % a % name);
	}
protected:
	template <typename T>
	void _write(const T & a) {
		m_file<<" "<<a<<" ";
	}
	bool checkFrequency(Frequency freq, const std::string & notation,
						char ignorePath = 0);
private:
    void writeByTypeAndLoc(int type, int loc, char * data);
	std::string fullPathName(const std::string & name);
private:
	std::ofstream m_file;
	std::vector<std::string > m_pathToName;
	static std::map<std::string, bool> VisitedPtr;
};
#endif        //  #ifndef BASELOG_H


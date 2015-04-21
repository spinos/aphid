/*
 *  HesperisFile.cpp
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisFile.h"
#include <AllHdf.h>
#include <HWorld.h>
#include <HCurveGroup.h>
#include <sstream>
HesperisFile::HesperisFile() {}
HesperisFile::HesperisFile(const char * name) : HFile(name) {}
HesperisFile::~HesperisFile() {}

bool HesperisFile::doWrite(const std::string & fileName)
{
	if(!HObject::FileIO.open(fileName.c_str(), HDocument::oReadAndWrite)) {
		setLatestError(BaseFile::FileNotWritable);
		return false;
	}
	
	HWorld grpWorld;
	grpWorld.save();
	std::stringstream sst;
	std::map<std::string, CurveGroup *>::iterator itcurve = m_curves.begin();
	for(; itcurve != m_curves.end(); ++itcurve) {
		sst.str("");
		sst<<"/world/"<<itcurve->first;
		HCurveGroup grpCurve(sst.str());
		grpCurve.save(itcurve->second);
		grpCurve.close();
	}
	
	grpWorld.close();
	HObject::FileIO.close();
	
	return true;
}

void HesperisFile::addCurve(const std::string & name, CurveGroup * data)
{ m_curves[name] = data; }

bool HesperisFile::doRead(const std::string & fileName)
{
	if(!HFile::doRead(fileName)) return false;
	
	std::cout<<" reading curves file "<<fileName<<"\n";
	HWorld grpWorld;
	grpWorld.load();
	
	bool allValid = true;
	std::stringstream sst;
	std::map<std::string, CurveGroup *>::iterator itcurve = m_curves.begin();
	for(; itcurve != m_curves.end(); ++itcurve) {
		sst.str("");
		sst<<"/world/"<<itcurve->first;
		HCurveGroup grpCurve(sst.str());
		if(!grpCurve.load(itcurve->second)) {
			std::cout<<" cannot load "<<sst.str();
			allValid = false;
		}
		
		grpCurve.close();
	}
	
	grpWorld.close();
	
	if(!allValid) {
		std::cout<<" encounter problem(s) reading the file.\n";
		return false;
	}
	
	std::cout<<" done reading curve file modified at "<<grpWorld.modifiedTimeStr()<<"\n";
	
	return true;
}

//:~
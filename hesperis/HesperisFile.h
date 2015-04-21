/*
 *  HesperisFile.h
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <HFile.h>
#include <string>
#include <map>
class CurveGroup;
class HesperisFile : public HFile {
public:
	HesperisFile();
	HesperisFile(const char * name);
	virtual ~HesperisFile();
	
	void addCurve(const std::string & name, CurveGroup * data);
	virtual bool doWrite(const std::string & fileName);
protected:

private:
	std::map<std::string, CurveGroup * > m_curves;
};
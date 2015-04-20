/*
 *  HesperisFile.h
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <HFile.h>
class HesperisFile : public HFile {
public:
	HesperisFile();
	HesperisFile(const char * name);
	virtual ~HesperisFile();
	
	virtual bool doWrite(const std::string & fileName);
protected:

private:
};
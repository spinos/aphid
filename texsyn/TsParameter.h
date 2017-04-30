/*
 *  TsParameter.h
 *  
 *
 *  Created by jian zhang on 9/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TS_PARAMETER_H
#define TS_PARAMETER_H

#include <LfParameter.h>

namespace tss {

class Parameter : public lfr::LfParameter {
	
public:
	Parameter(int argc, char *argv[]);
	virtual ~Parameter();
	
protected:
	virtual void printVersion() const;
	virtual void printUsage() const;
	virtual void printDescription() const;
	
};

}

#endif
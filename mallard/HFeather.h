/*
 *  HFeather.h
 *  mallard
 *
 *  Created by jian zhang on 9/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HBase.h>
class MlFeather;
class HFeather : public HBase {
public:
	HFeather(const std::string & path);
	
	virtual char save(MlFeather * feather);
	virtual char load(MlFeather * feather);
	int loadId();
private:
};
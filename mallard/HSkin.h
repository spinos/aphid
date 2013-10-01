/*
 *  HSkin.h
 *  mallard
 *
 *  Created by jian zhang on 10/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <HBase.h>
class MlSkin;
class HSkin : public HBase {
public:
	HSkin(const std::string & path);
	
	virtual char save(MlSkin * s);
	virtual char load(MlSkin * s);
private:
};
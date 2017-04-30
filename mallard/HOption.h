/*
 *  HOption.h
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HBase.h>
class RenderOptions;
class HOption : public HBase {
public:
	HOption(const std::string & path);
	virtual ~HOption();
	
	virtual char save(RenderOptions * opt);
	virtual char load(RenderOptions * opt);
private:
	
};
#ifndef HFRAMERANGE_H
#define HFRAMERANGE_H

/*
 *  HFrameRange.h
 *  testbcc
 *
 *  Created by jian zhang on 4/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <HBase.h>
class AFrameRange;
class HFrameRange : public HBase {
public:
	HFrameRange(const std::string & path);
	virtual ~HFrameRange();
	
	virtual char verifyType();
	virtual char save(AFrameRange * fr);
	virtual char load(AFrameRange * fr);
	
private:
	
};
#endif        //  #ifndef HFRAMERANGE_H


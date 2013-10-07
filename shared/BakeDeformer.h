/*
 *  BakeDeformer.h
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseFile.h>
#include <BaseDeformer.h>
class HBase;
class BakeDeformer : public BaseFile, public BaseDeformer {
public:
	BakeDeformer();
	virtual ~BakeDeformer();
	
	virtual char load(const char * filename);
private:
	bool findMatchedMesh(HBase & grp);
	bool isGeomMatched(HBase & grp);
	bool isBakeMatched(HBase & grp);
	void processFrameRange();
private:
	std::string m_bakePath;
	int m_minFrame, m_maxFrame;
};
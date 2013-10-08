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
#include <AllMath.h>
#include <map>
class HBase;
class BakeDeformer : public BaseFile, public BaseDeformer {
public:
	BakeDeformer();
	virtual ~BakeDeformer();
	
	virtual char load(const char * filename);
	virtual char solve();
	
	void setCurrentFrame(int x);
	
	int minFrame() const;
	int maxFrame() const;
private:
	bool findMatchedMesh(HBase & grp);
	bool isGeomMatched(HBase & grp);
	bool isBakeMatched(HBase & grp);
	void processFrameRange();
	void processFrameCenters();
	void clearFrames();
private:
	std::string m_bakePath;
	std::map<int, Vector3F> m_frameCenters;
	int m_minFrame, m_maxFrame, m_currentFrame;
};
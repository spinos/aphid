/*
 *  BakeDeformer.h
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HRFile.h>
#include <BaseDeformer.h>
#include <AllMath.h>
#include <map>
class HBase;
class BakeDeformer : public HRFile, public BaseDeformer {
public:
	BakeDeformer();
	virtual ~BakeDeformer();
	
	virtual char solve();
	
	virtual bool doRead(const std::string & filename);
	
	void setCurrentFrame(int x);
	
	int minFrame() const;
	int maxFrame() const;
	
	void verbose() const;
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
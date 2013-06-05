/*
 *  YarnPatch.h
 *  knitfabric
 *
 *  Created by jian zhang on 6/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <accPatch.h>

class YarnPatch : public AccPatch {
public:
	YarnPatch();
	void setQuadVertices(unsigned *v);
	void findWaleEdge(unsigned v0, unsigned v1);
	void waleEdges(short &n, unsigned * v) const;
private:
	void setWaleEdge(short i, short j);
	void setSecondWaleEdge(short j, char dir);
	unsigned * m_quadVertices;
	short m_waleVertices[4];
	short m_numWaleEdges;
};
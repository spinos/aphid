/*
 *  DrawBvh.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/18/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
class GeoDrawer;
class CudaLinearBvh;
class BaseBuffer;
class CudaBroadphase;
class DrawBvh {
public:
	DrawBvh();
	virtual ~DrawBvh();
	
	void setDrawer(GeoDrawer * drawer);
	void setBvh(CudaLinearBvh * bvh);
	void setBroadphase(CudaBroadphase * broadphase);
	void addDispalyLevel();
	void minusDispalyLevel();
	
	void bound();
	void leaf();
	void hash();
	void hierarch();
	
	void printHash();
	void printPairCounts();
private:
	GeoDrawer * m_drawer;
	CudaLinearBvh * m_bvh;
	CudaBroadphase * m_broadphase;
	BaseBuffer * m_displayLeafAabbs;
	BaseBuffer * m_displayInternalAabbs;
	BaseBuffer * m_displayInternalDistance;
	BaseBuffer * m_displayLeafHash;
	BaseBuffer * m_internalChildIndices;
	BaseBuffer * m_pairCounts;
	int m_hostRootNodeInd;
	int m_displayLevel;
};
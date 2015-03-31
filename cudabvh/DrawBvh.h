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
	void addDispalyLevel();
	void minusDispalyLevel();
	
	void leaf();
	
	void printHash();
	
	void printPairCounts(CudaBroadphase * broadphase);
	
	void showOverlappingPairs(CudaBroadphase * broadphase);
	void showBound(CudaLinearBvh * bvh);
	void showHash(CudaLinearBvh * bvh);
	void showHierarch(CudaLinearBvh * bvh);
	
private:
	GeoDrawer * m_drawer;
	CudaLinearBvh * m_bvh;
	BaseBuffer * m_displayLeafAabbs;
	BaseBuffer * m_displayInternalAabbs;
	BaseBuffer * m_displayInternalDistance;
	BaseBuffer * m_displayLeafHash;
	BaseBuffer * m_internalChildIndices;
	BaseBuffer * m_pairCounts;
	BaseBuffer * m_scanCounts;
	BaseBuffer * m_pairCache;
	BaseBuffer * m_boxes;
	BaseBuffer * m_uniquePairs;
	BaseBuffer * m_scanUniquePairs;
	int m_hostRootNodeInd;
	int m_displayLevel;
};
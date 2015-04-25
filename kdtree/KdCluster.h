/*
 *  KdCluster.h
 *  testkdtree
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <KdTree.h>
class GeometryArray;
class KdCluster : public KdTree {
public:
	KdCluster();
	virtual ~KdCluster();
	
	const unsigned numGroups() const;
	GeometryArray * group(unsigned idx) const;
	
	virtual void create();
	virtual void rebuild();
protected:
	virtual void clear();
private:
	void recursiveFindGroup(KdTreeNode *node, const BoundingBox & box);
	void leafWriteGroup(KdTreeNode *node, const BoundingBox & box);
	void clearGroups();
private:
	GeometryArray ** m_groupGeometries;
	unsigned m_currentGroup;
};
#ifndef KDCLUSTER_H
#define KDCLUSTER_H

/*
 *  KdCluster.h
 *  testkdtree
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <KdTree.h>
#include <map>
class GeometryArray;
class KdCluster : public KdTree {
public:
	KdCluster();
	virtual ~KdCluster();
	
	const unsigned numGroups() const;
	const unsigned currentGroup() const;
	GeometryArray * group(unsigned idx) const;
    void setGroupGeometry(unsigned idx, GeometryArray * geos);
	
	virtual void create();
	virtual void rebuild();
	
	virtual bool intersectRay(const Ray * eyeRay);
	void setCurrentGroup(unsigned x);
    
    bool isGroupIdValid(unsigned x) const;
protected:
	virtual void clear();
private:
	void recursiveFindGroup(KdTreeNode *node, const BoundingBox & box);
	void leafWriteGroup(KdTreeNode *node, const BoundingBox & box);
	void clearGroups();
	
	bool recursiveIntersectRay(KdTreeNode *node, const Ray * eyeRay, const BoundingBox & box);
	bool leafIntersectRay(KdTreeNode *node, const Ray * eyeRay);
private:
	GeometryArray ** m_groupGeometries;
	std::map<KdTreeNode *, unsigned > m_nodeGroupInd;
	unsigned m_currentGroup;
};
#endif        //  #ifndef KDCLUSTER_H

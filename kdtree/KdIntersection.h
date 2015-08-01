#ifndef KDINTERSECTION_H
#define KDINTERSECTION_H

/*
 *  KdIntersection.h
 *  testkdtree
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <KdTree.h>
class GeometryArray;
class KdIntersection : public KdTree {
public:
	KdIntersection();
	virtual ~KdIntersection();
	
	virtual bool intersectBox(const BoundingBox & box);
	virtual bool intersectTetrahedron(const Vector3F * tet);
    virtual unsigned countElementIntersectBox(const BoundingBox & box);
	
protected:
	
private:
	bool recursiveIntersectBox(KdTreeNode *node, const BoundingBox & box);
	bool leafIntersectBox(KdTreeNode *node, const BoundingBox & box);
	bool recursiveIntersectTetrahedron(KdTreeNode *node, const BoundingBox & box);
	bool leafIntersectTetrahedron(KdTreeNode *node, const BoundingBox & box);
	void internalCountElementIntersectBox(unsigned & result, KdTreeNode *node, const BoundingBox & box);
    void leafCountElementIntersectBox(unsigned & result, KdTreeNode *node, const BoundingBox & box);
private:
	BoundingBox m_testBox;
	Vector3F m_testTetrahedron[4];
};
#endif        //  #ifndef KDINTERSECTION_H

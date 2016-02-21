/*
 *  SceneContainer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <IntersectionContext.h>
#include <SelectionContext.h>
#include <Geometry.h>

class RandomMesh;
class KdTreeDrawer;
class BezierCurve;
class KdCluster;
class KdTree;
class GeometryArray;
class Ray;
#define NUM_MESHES 201
class SceneContainer {
	IntersectionContext m_intersectCtx;
	SelectionContext m_selectCtx;
	Geometry::ClosestToPointTestResult m_closestPointTest;
	
public:
	SceneContainer(KdTreeDrawer * drawer);
	virtual ~SceneContainer();
	
	void renderWorld();
	void upLevel();
	void downLevel();
	void intersect(const Ray * incident);
	
protected:

private:
	void testMesh();
	void testCurve();
	void drawIntersection();
	void drawClosest();
	
private:
	KdTreeDrawer * m_drawer;
	RandomMesh * m_mesh[NUM_MESHES];
	GeometryArray * m_curves;
	KdCluster * m_cluster;
	KdTree * m_tree;
	int m_level;
};
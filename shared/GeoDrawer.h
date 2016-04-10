#ifndef GEODRAWER_H
#define GEODRAWER_H

/*
 *  GeoDrawer.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "MeshDrawer.h"

namespace aphid {

class GeodesicSphereMesh;
class PyramidMesh;
class CubeMesh;
class CircleCurve;
class Primitive;
class DiscMesh;
class Geometry;
class GeometryArray;
class ATetrahedronMesh;
class AOrientedBox;
//class CartesianGrid;
//class APointCloud;
//class SelectionArray;
//class BaseTransform;
//class TransformManipulator;
//class SkeletonJoint;
//class Anchor;
//class SpaceHandle;

class GeoDrawer : public MeshDrawer {
public:
	GeoDrawer();
	virtual ~GeoDrawer();
	
	void unitBoxAt(const Vector3F & p, const float & size) const;
	void unitCubeAt(const Vector3F & p, const float & size) const;
	void box(float width, float height, float depth);
	
	void sphere(float size = 1.f) const;
	void cube(const Vector3F & p, const float & size) const;
	void solidCube(float x, float y, float z, float size);
	
	void circleAt(const Vector3F & pos, const Vector3F & nor);
	void circleAt(const Matrix44F & mat, float radius);
	
	void arrow0(const Vector3F & at, const Vector3F & dir, float l, float w) const;
	void arrow2(const Vector3F& origin, const Vector3F& dest, float width) const;
	void arrow(const Vector3F& origin, const Vector3F& dest) const;
	
	void coordsys(float scale = 1.f) const;
	void coordsys(const Vector3F & scale) const;
	void coordsys(const Matrix33F & orient, float size = 1.f, Vector3F * p = 0);
	void coordsys(const Matrix33F & orient, const Vector3F & p, 
					const Vector3F & size = Vector3F(1.f, 1.f, 1.f) );
	
	//void manipulator(TransformManipulator * m);
	//void spaceHandle(SpaceHandle * hand);
	//void anchor(Anchor *a, char active = 0);
	
	//void transform(BaseTransform * t) const;
	
	//void skeletonJoint(SkeletonJoint * joint);
	//void moveHandle(int axis, bool active) const;
	//void spinHandle(TransformManipulator * m, bool active) const;
	//void spinPlanes(BaseTransform * t) const;
	
	//void components(SelectionArray * arr);
	
	void primitive(Primitive * prim);
	
	void drawDisc(float scale = 1.f) const;
	void drawSquare(const BoundingRectangle & b) const;
	void aabb(const Vector3F & low, const Vector3F & high) const;
	void tetrahedron(const Vector3F * p) const;
	
	void geometry(Geometry * geo) const;
	void geometryArray(GeometryArray * arr) const;
	
	void setAlignDir(const Vector3F & v);
	void alignedDisc(const Vector3F & pos, float radius) const;
	void alignedCircle(const Vector3F & pos, float radius) const;
	
	//void pointCloud(APointCloud * cloud) const;
	void tetrahedronMesh(ATetrahedronMesh * mesh) const;
	//void cartesianGrid(CartesianGrid * grid) const;
	
	void orientedBox(const AOrientedBox * ob);
	
private:
	Vector3F m_alignDir;
	GeodesicSphereMesh * m_sphere;
	PyramidMesh * m_pyramid;
	CubeMesh * m_cube;
	CircleCurve * m_circle;
	DiscMesh * m_disc;
	Vector3F * m_boxVBuf;
	float m_spaceBuf[16];
};

}
#endif        //  #ifndef GEODRAWER_H

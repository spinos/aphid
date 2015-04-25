/*
 *  MeshDrawer.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "LineDrawer.h"
#include <deque>
class TriangleMesh;
class BaseMesh;
class BaseDeformer;
class BaseField;
class MeshDrawer : public LineDrawer {
public:
	MeshDrawer();
	virtual ~MeshDrawer();
	void triangleMesh(const TriangleMesh * mesh, const BaseDeformer * deformer = 0) const;
	void quadMesh(const BaseMesh * mesh) const;
	void drawMesh(const BaseMesh * mesh, const BaseDeformer * deformer = 0) const;
	void drawPolygons(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void drawPoints(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void showNormal(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void edge(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void hiddenLine(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void triangle(const BaseMesh * mesh, unsigned idx);
	void patch(const BaseMesh * mesh, unsigned idx);
	void patch(const BaseMesh * mesh, const std::deque<unsigned> & sel) const;
	void perVertexVector(BaseMesh * mesh, const std::string & name);
	void vertexNormal(BaseMesh * mesh);
	void tangentFrame(const BaseMesh * mesh, const BaseDeformer * deformer = 0);
	void field(const BaseField * f);
};
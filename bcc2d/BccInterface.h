/*
 *  BccInterface.h
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <HesperisInterface.h>
#include <AOrientedBox.h>
class KdTreeDrawer;
class AGenericMesh;
class ATetrahedronMesh;
class ATetrahedronMeshGroup;
class ATriangleMesh;
class BccWorld;
class AdaptiveField;
class BaseBuffer;
class MeshSeparator;

class BccInterface : public HesperisInterface {
public:
	BccInterface();
	virtual ~BccInterface();
	
	bool createWorld(BccWorld * world);
	void drawWorld(BccWorld * world, KdTreeDrawer * drawer);
	bool saveWorld(BccWorld * world);
	
	bool loadTriangleGeometry(BccWorld * world, const std::string & filename);
	bool loadCurveGeometry(BccWorld * world, const std::string & filename);
	bool loadPatchGeometry(BccWorld * world, const std::string & filename);
protected:
	bool separate(ATriangleMesh * mesh, std::vector<AOrientedBox> & patchBoxes);
private:
    void drawTetrahedronMesh(ATetrahedronMesh * m, KdTreeDrawer * drawer);
	void drawGeometry(GeometryArray * geos, KdTreeDrawer * drawer);
	void drawAnchors(AGenericMesh * mesh, KdTreeDrawer * drawer,
						float drawSize);
    void testBlockMesh();
private:
	ATriangleMesh * m_patchMesh;
    ATetrahedronMeshGroup * m_tetMesh;
	MeshSeparator * m_patchSeparator;
};
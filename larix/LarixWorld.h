/*
 *  LarixWorld.h
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
class ATetrahedronMesh;
class APointCloud;
class AdaptiveField;

class LarixWorld {
public:
	LarixWorld();
	virtual ~LarixWorld();
    
    void setTetrahedronMesh(ATetrahedronMesh * m);
    ATetrahedronMesh * tetrahedronMesh() const;
	
	void setPointCloud(APointCloud * pc);
	APointCloud * pointCloud() const;
	
	void setField(AdaptiveField * g);
	AdaptiveField * field() const;

protected:

private:
    ATetrahedronMesh * m_mesh;
	APointCloud * m_cloud;
	AdaptiveField * m_field;
};
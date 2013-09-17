/*
 *  MlTessellate.h
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
class AccPatchMesh;
class MeshTopology;
class MlFeather;
class MlTessellate {
public:
	MlTessellate();
	void setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo);
	void setFeather(MlFeather * feather);
private:
	AccPatchMesh * m_body;
	MeshTopology * m_topo;
	MlFeather * m_feather;
};
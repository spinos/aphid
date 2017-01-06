/*
 *  DrawAvianArm.h
 *  cinchona
 *
 *  Created by jian zhang on 1/1/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DRAW_AVIAN_ARM_H
#define DRAW_AVIAN_ARM_H
#include <ogl/DrawArrow.h>
#include "AvianArm.h"

class Ligament;
class FeatherMesh;
class FeatherDeformer;

class DrawAvianArm : public AvianArm, public aphid::DrawArrow {

public:
	DrawAvianArm();
	virtual ~DrawAvianArm();
	
protected:
	void drawSkeletonCoordinates();
	void drawLigaments();
	void drawFeathers();
	void drawFeatherOrietations();
	void drawFeatherLeadingEdges();
	void drawRibs();
	
private:
	void drawLigament(const Ligament & lig);
	void drawFeatherMesh(const FeatherMesh * mesh,
						const FeatherDeformer * deformer);
	void drawFeatherLeadingEdge(const FeatherMesh * mesh,
						const FeatherDeformer * deformer);
	
};
#endif

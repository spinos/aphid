/*
 *  DeformationAnalysisDrawer.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "DeformationAnalysisDrawer.h"

DeformationAnalysisDrawer::DeformationAnalysisDrawer() {}
DeformationAnalysisDrawer::~DeformationAnalysisDrawer() {}

void DeformationAnalysisDrawer::visualize(DeformationAnalysis * analysis)
{
	setWired(1);
	setColor(0.f, 1.f, 0.7f);
	drawMesh(analysis->getMeshA());
	setColor(0.f, .7f, 1.f);
	drawMesh(analysis->getMeshB());
}
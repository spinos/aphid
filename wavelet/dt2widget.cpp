/*
 *  dt2widget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <Calculus.h>
#include "dt2widget.h"
#include "dtdwt2.h"
#include "gensig.h"

using namespace aphid;

Dt2Widget::Dt2Widget(QWidget *parent) : Plot2DWidget(parent)
{
/// example signal
	UniformPlot2DImage * Xp = new UniformPlot2DImage;
	Xp->create(256, 256, 1);
	
	gen2dsig(Xp, 256, 256, 1);
	gen2dsigFrac(Xp, 256, 256, 1, 0, 0, 256, 256);
	Xp->setDrawScale(1.f);
	Xp->updateImage();
	
	addImage(Xp);
	
	wla::DualTree2 tree;
	tree.analize(Xp->data(), 4);
	
	std::cout<<"\n last stage "<<tree.lastStage();
	
	int i, j, k;
#if 0	
#if 1
	for(i=0;i<=tree.lastStage();++i) {
#else
	i=tree.lastStage();	
#endif
		const int n = (i==tree.lastStage() ) ? 1 : 3;
		
		for(j=0;j<2;++j) {
			
			for(k=0;k<n;++k) {
				std::cout<<"\n "<<i<<" "<<j<<" "<<k;
				
				const Array3<float> & stg = tree.stageBand(i, j, k);
				
				UniformPlot2DImage * wp = new UniformPlot2DImage;
				wp->create(stg);
				
				wp->setDrawScale(1.f);					
				wp->updateImage(i!=tree.lastStage());
				addImage(wp);
			}
		}
#if 1
	}
#endif
#endif

	Array3<float> sythy;
	tree.synthesize(sythy);
	UniformPlot2DImage * yp = new UniformPlot2DImage;
	yp->create(sythy);
	yp->setDrawScale(1.f);					
	yp->updateImage();
	//addImage(yp);
	
	float mxe = 0.f;
	
	for(k=0;k<1;++k) {
		Xp->channel(k)->maxAbsError(mxe, *sythy.rank(k) );
		
	}
	
	std::cout<<"\n max err "<<mxe;
	
/// sythesis signal
	UniformPlot2DImage * Xs = new UniformPlot2DImage;
	Xs->create(256, 256, 1);
	
	gen2dsig(Xs, 256, 256, 1, .1f);
	gen2dsigFrac(Xs, 256, 256, 1, 96, 96, 160, 160);
	gen2dsigFrac(Xs, 256, 256, 1, 96, 96, 160, 160);
	Xs->setDrawScale(1.f);
	Xs->updateImage();
	
	addImage(Xs);
	
/// synthesis tree
	wla::DualTree2 treeS;
	treeS.analize(Xs->data(), 4);
	const int lsm = treeS.lastStageBand(0).numRows();
	const int lsn = treeS.lastStageBand(0).numCols();
	std::cout<<"\n last stage m "<<lsm<<" n "<<lsn;
	
	tree.nns(treeS, 8, 8);
	
	std::cout.flush();	
}

Dt2Widget::~Dt2Widget()
{}


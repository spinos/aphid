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
	UniformPlot2DImage * Xp = new UniformPlot2DImage;
	gen2dsig(Xp, 256, 256, 1);
	Xp->setDrawScale(1.f);
	Xp->updateImage();
	
	addImage(Xp);
	
	wla::DualTree2 tree;
	tree.analize(Xp->data(), 1);
	
	std::cout<<"\n last stage "<<tree.lastStage();
	
	int i, j, k;
	
#if 0
	for(i=0;i<=tree.lastStage();++i) {
#else
	i=tree.lastStage();	
#endif
		const int n = (i==tree.lastStage() ) ? 1 : 3;
		
		for(j=0;j<2;++j) {
			
			for(k=0;k<n;++k) {
				std::cout<<"\n "<<i<<" "<<j<<" "<<k;
				
				const Array3<float> & stg = tree.stage(i, j, k);
				
				UniformPlot2DImage * wp = new UniformPlot2DImage;
				wp->create(stg);
				
				wp->setDrawScale(1.f);					
				wp->updateImage(i!=tree.lastStage());
				addImage(wp);
			}
		}
#if 0
	}
#endif

	Array3<float> sythy;
	tree.synthesize(sythy);
	UniformPlot2DImage * yp = new UniformPlot2DImage;
	yp->create(sythy);
	yp->setDrawScale(1.f);					
	yp->updateImage();
	addImage(yp);
	
	float mxe = 0.f;
	
	for(k=0;k<1;++k) {
		Xp->channel(k)->maxAbsError(mxe, *sythy.rank(k) );
		
	}
	
	std::cout<<"\n max err "<<mxe;
	
	std::cout.flush();	
}

Dt2Widget::~Dt2Widget()
{}


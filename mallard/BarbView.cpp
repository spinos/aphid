#include "BarbView.h"
#include <QtGui>
#include <MlFeather.h>
#include <MlFeatherCollection.h>
#include <KdTreeDrawer.h>
#include <PseudoNoise.h>
#include <BaseCamera.h>
#include "MlVane.h"
#include <AdaptableStripeBuffer.h>

BarbView::BarbView(QWidget *parent) : Base3DView(parent)
{
	std::cout<<" Barbview ";
}

BarbView::~BarbView() {}

void BarbView::clientDraw()
{
   MlFeather *f = selectedExample();
	if(!f) return;
	
	getDrawer()->stripes(f->stripe(), getCamera()->eyeDirection());
}

void BarbView::clientSelect() {}

void BarbView::clientMouseInput() {}

void BarbView::receiveShapeChanged()
{
	MlFeather *f = selectedExample();
	if(!f) return;
	
	float * dst = f->angles();
	const short ns = f->numSegment();
	for(short s=0; s < ns; s++) {
		dst[s] = -0.4f * s / (float)ns;
	}
	f->bend();
	f->testVane();
	
	f->sampleColor(f->levelOfDetail());
	f->samplePosition(f->levelOfDetail());

	update();
}

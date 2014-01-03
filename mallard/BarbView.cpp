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
	perspCamera()->setNearClipPlane(1.f);
	perspCamera()->setFarClipPlane(100000.f);
	usePerspCamera();
}

BarbView::~BarbView() {}

void BarbView::clientDraw()
{
   MlFeather *f = selectedExample();
	if(!f) return;
	
	getDrawer()->stripes(f->stripe(), getCamera()->eyeDirection());
}

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
	
	const Vector3F rt(4.f, 0.f, 4.f);
	const float rd = f->scaledShaftLength();
	
	const Vector3F eye = getCamera()->eyePosition();
	const float fov = getCamera()->fieldOfView();
	setEyePosition(eye);
	setFieldOfView(fov);
	float lod = computeLOD(rt, rd, f->numSegment() * 8);
	f->computeNoise();
	f->sampleColor(lod);
	f->samplePosition(lod);

	update();
}

void BarbView::clientSelect() {}

void BarbView::clientDeselect()
{
	receiveShapeChanged();
}

void BarbView::receiveLodChanged(double l)
{
	setOverall(l);
	receiveShapeChanged();
}

#include "BarbView.h"
#include <QtGui>
#include <MlFeather.h>
#include <MlFeatherCollection.h>
#include <KdTreeDrawer.h>
#include <PseudoNoise.h>
#include <BaseCamera.h>
#include "MlVane.h"
#include <AdaptableStripeBuffer.h>
#include <FractalPlot.h>

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

void BarbView::test()
{
	MlFeather *f = selectedExample();
	if(!f) return;
	FractalPlot plot;
	plot.createPlot(1024);
	plot.computePlot(f->seed());
	std::vector<Vector3F> pts;
	float d = 1.f/512.f;
	float noi;
	Vector3F pt(0.f, 0.f, 1.f);
	for(unsigned i=0; i < 512; i++) {
		pts.push_back(pt);
		noi = plot.getNoise(d*i, 64, overall(), f->seed());
		pt = Vector3F(d * i * 128.f, noi * 10.f, 1.f);
		pts.push_back(pt);
	}
	
	getDrawer()->lines(pts);
}

void BarbView::focusInEvent(QFocusEvent * event)
{
	receiveShapeChanged();
	Base3DView::focusInEvent(event);
}

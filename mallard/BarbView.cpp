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
	
	m_f = f;
	boost::thread t(boost::bind(&BarbView::sampleShape, this));
	t.join();
}

void BarbView::sampleShape()
{
    io_mutex.lock();
	
    m_f->setResShaft(m_resShaft);
    m_f->setResBarb(m_resBarb);
    m_f->setNumSeparate(m_numSeparate);
    m_f->setSeed(m_seed);
    m_f->setFuzzy(m_fuzzy);
    m_f->setSeparateStrength(m_separateStrength);
	m_f->m_barbShrink = m_barbShrink;
	m_f->m_shaftShrink = m_shaftShrink;

	float * dst = m_f->angles();
	const short ns = m_f->numSegment();
	for(short s=0; s < ns; s++) {
		dst[s] = -0.4f * s / (float)ns;
	}
	m_f->bend();
	m_f->testVane();
	
	const Vector3F rt(4.f, 0.f, 4.f);
	const float rd = m_f->scaledShaftLength();
	
	const Vector3F eye = getCamera()->eyePosition();
	const float fov = getCamera()->fieldOfView();
	setEyePosition(eye);
	setFieldOfView(fov);
	float lod = computeLOD(rt, rd, m_f->numSegment() * 8);
	m_f->computeNoise();
	m_f->sampleColor(lod);
	m_f->samplePosition(lod);
	
	update();
	io_mutex.unlock();
}

void BarbView::clientSelect() {}

void BarbView::clientDeselect()
{
}

void BarbView::receiveLod(double l)
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

void BarbView::receiveSeed(int s)
{
    m_seed = s;
}

void BarbView::receiveNumSeparate(int n)
{
    m_numSeparate = n;
}

void BarbView::receiveSeparateStrength(double k)
{
    m_separateStrength = k;
}

void BarbView::receiveFuzzy(double f)
{
    m_fuzzy = f;
}

void BarbView::receiveResShaft(int g)
{
    m_resShaft = g;
}

void BarbView::receiveResBarb(int g)
{
    m_resBarb = g;
}

void BarbView::receiveBarbShrink(double x)
{
	m_barbShrink = x;
}

void BarbView::receiveShaftShrink(double x)
{
	m_shaftShrink = x;
}
/*
 *   dthfwidget.cpp
 *
 */
 
#include <QtGui>
#include <math/Calculus.h>
#include "gauwidget.h"
#include <img/HeightField.h>
#include <img/ExrImage.h>
#include <img/ImageSensor.h>

using namespace aphid;

GauWidget::GauWidget(const ExrImage * img,
					QWidget *parent) : Plot2DWidget(parent)
{
	setMargin(0, 0);
	
	if(!img->isValid() ) {
		return;
	}
	
	Array3<float> inputX;
	img->sampleRed(inputX );
	
	m_gau = new img::HeightField;
	m_gau->create(inputX);
	m_gau->setRange(1024);
	m_gau->verbose();
	const int & n = m_gau->numLevels();
#if 0
	for(int i=0;i<n;++i) {
		UniformPlot2DImage * plot = new UniformPlot2DImage;
		plot->setDrawScale(1.f);
		plot->create(m_gau->levelSignal(i) );
		plot->updateImage();
		addImage(plot);
	}
#endif

	m_Y = new Array3<float>();
	
	m_plotY = new UniformPlot2DImage;
	m_plotY->setDrawScale(1.f);
	
	addImage(m_plotY);
	
	for(int i=0;i<n;++i) {
		UniformPlot2DImage * plotDev = new UniformPlot2DImage;
	
		plotDev->create(m_gau->levelDerivative(i) );
		plotDev->updateImage(true);
	
		addImage(plotDev);
	}
}

GauWidget::~GauWidget()
{}

void GauWidget::resizeEvent(QResizeEvent * event)
{
	BaseImageWidget::resizeEvent(event);
	resample();
		
}

void GauWidget::resample()
{
	int n = portSize().width();
	int m = n * m_gau->aspectRatio();
	
	m_Y->create(m, n, 1);
	
	img::ImageSensor<img::HeightField> sensor(Vector2F(0,0),
		Vector2F(1024,0), n,
		Vector2F(0,1024.f * m_gau->aspectRatio() ), m);
	sensor.verbose();
		
	sensor.sense(m_Y, 0, *m_gau);
	
	m_plotY->create(*m_Y);
	m_plotY->updateImage();
}

/*
 *   dthfwidget.cpp
 *
 */
 
#include <QtGui>
#include <math/Calculus.h>
#include "gauwidget.h"
#include <img/HeightField.h>
#include <img/ExrImage.h>

using namespace aphid;

GauWidget::GauWidget(const ExrImage * img,
					QWidget *parent) : Plot2DWidget(parent)
{
	setMargin(0, 0);
	
	if(!img->isValid() ) {
		return;
	}
	
	Array3<float> inputX;
	//inputX.create(xdim, ydim, 1);
	//img->sampleRed(inputX.rank(0)->v() );
	img->sampleRed(inputX );
	
	m_gau = new img::HeightField(inputX);
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
	int m = portSize().height();
	int n = m / m_gau->aspectRatio();
	float du = 1.f / (float)n;
	float dv = 1.f / (float)m;
	float hdu = du * .5f;
	float hdv = dv * .5f;
	
	m_Y->create(m, n, 1);
	
	Array2<float> * slice = m_Y->rank(0);
	
	float filterSize = (float)m_gau->levelSignal(0).numRows() / (float)m;
	
	img::BoxSampleProfile<float> sampler;
	sampler._channel = 0;
	m_gau->getSampleProfle(&sampler, filterSize);
	
	for(int j=0;j<n;++j) {
		float * colj = slice->column(j);
		
		sampler._uCoord = hdu + du * j;
		
		for(int i=0;i<m;++i) {
		
			sampler._vCoord = hdv + dv * i;
			
			colj[i] = m_gau->sample(&sampler);
		}
	}
	
	m_plotY->create(*m_Y);
	m_plotY->updateImage();
}

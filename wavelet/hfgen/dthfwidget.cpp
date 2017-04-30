/*
 *   dthfwidget.cpp
 *
 */
 
#include <QtGui>
#include <math/Calculus.h>
#include "dthfwidget.h"
#include <wla/dtdwt2.h>
#include <img/ExrImage.h>

using namespace aphid;

DthfWidget::DthfWidget(const ExrImage * img,
					QWidget *parent) : Plot2DWidget(parent)
{
	setMargin(0, 0);
	
	setMinimumWidth(512);
	setMinimumHeight(512);
	
	m_levelScale[0] = 1.f;
	m_levelScale[1] = 1.f;
	m_levelScale[2] = 1.f;
	m_levelScale[3] = 1.f;
	
	if(!img->isValid() ) {
		return;
	}
	
	int xdim = img->getWidth();
	int ydim = img->getHeight();

	m_inputX = new Array3<float>();
	m_inputX->create(xdim, ydim, 1);
	img->sampleRed(m_inputX->rank(0)->v() );
	
	int level = 4;
	std::cout<<"\n dtdwt level "<<level;

	m_synthesis = new wla::DualTree2;
	
	m_plot = new UniformPlot2DImage;
	m_plot->setDrawScale(1.f);
	addImage(m_plot);
	
}

void DthfWidget::checkSynthesisErr(const Array3<float> & synth)
{
	float mxe = 0.f;
	m_inputX->rank(0)->maxAbsError(mxe, *(synth.rank(0)) );
	
	std::cout<<"\n max err "<<mxe;
	std::cout.flush();
}

DthfWidget::~DthfWidget()
{}

void DthfWidget::recvL0scale(double x)
{
	m_levelScale[0] = x;
	resynthsize();
}

void DthfWidget::recvL1scale(double x)
{
	m_levelScale[1] = x;
	resynthsize();
}

void DthfWidget::recvL2scale(double x)
{
	m_levelScale[2] = x;
	resynthsize();
}

void DthfWidget::recvL3scale(double x)
{
	m_levelScale[3] = x;
	resynthsize();
}

void DthfWidget::resizeEvent(QResizeEvent * event)
{
	BaseImageWidget::resizeEvent(event);
	resynthsize();
		
}

void DthfWidget::resynthsize()
{
	int pw = portSize().width();
	int ph = portSize().height();
	if(pw > m_inputX->numRows() ) {
		pw = m_inputX->numRows();
	}
	if(ph > m_inputX->numCols() ) {
		ph = m_inputX->numCols();
	}
	
	Array3<float> wndX;
	wndX.create(pw, ph, 1);
	
	Array2SampleProfile<float> sampler;
	sampler._translate = Float2(0.3191f, 0.432f);
	sampler._scale = Float2(.2743f, .2743f);
	sampler._defaultValue = 0.5f;
	
	wndX.sample(*m_inputX, &sampler);
	
	m_synthesis->analize(wndX, 4);
	
	m_synthesis->scaleUp(0, m_levelScale[0]);
	m_synthesis->scaleUp(1, m_levelScale[1]);
	m_synthesis->scaleUp(2, m_levelScale[2]);
	m_synthesis->scaleUp(3, m_levelScale[3]);
	
	Array3<float> wndY;
	m_synthesis->synthesize(wndY);
	
	m_plot->create(wndY);
	m_plot->updateImage();
	update();
}
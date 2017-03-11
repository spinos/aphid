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
	setMargin(4, 4);
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

	wla::DualTree2 tree;
	tree.analize(*m_inputX, level);
	
	Array3<float> sythy;
	tree.synthesize(sythy);
	checkSynthesisErr(sythy);
	
	m_plot = new UniformPlot2DImage;
	m_plot->create(sythy);
	m_plot->setDrawScale(1.f);					
	m_plot->updateImage();
	
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
	resynth();
}

void DthfWidget::recvL1scale(double x)
{
	m_levelScale[1] = x;
	resynth();
}

void DthfWidget::recvL2scale(double x)
{
	m_levelScale[2] = x;
	resynth();
}

void DthfWidget::recvL3scale(double x)
{
	m_levelScale[3] = x;
	resynth();
}

void DthfWidget::resynth()
{
	wla::DualTree2 tree;
	tree.analize(*m_inputX, 4);
	
	tree.scaleUp(0, m_levelScale[0]);
	tree.scaleUp(1, m_levelScale[1]);
	tree.scaleUp(2, m_levelScale[2]);
	tree.scaleUp(3, m_levelScale[3]);
	
	Array3<float> sythy;
	tree.synthesize(sythy);
	
	checkSynthesisErr(sythy);
	
	m_plot->create(sythy);
	m_plot->updateImage();
	update();
}

#include <QtGui>
#include "paramdialog.h"
#include "paramwidget.h"

using namespace aphid;

ParamDialog::ParamDialog(QWidget *parent)
    : QDialog(parent)
{	
	m_wig = new ParamWidget(this);
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_wig);
	layout->setStretch(2, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    setWindowTitle(tr("Airfoil Parameters") );
    resize(480, 480);
	
	connect(m_wig, SIGNAL(camberChanged(double)), this, SLOT(recvCamber(double)));
    connect(m_wig, SIGNAL(positionChanged(double)), this, SLOT(recvPosition(double)));
    connect(m_wig, SIGNAL(thicknessChanged(double)), this, SLOT(recvThickness(double)));
    
	m_cpt.set(0.02f, 0.4f, 0.15f);
}

void ParamDialog::recvCamber(double x)
{
	m_cpt.x = x * 0.01;
	emit paramChanged(m_cpt); 
}

void ParamDialog::recvPosition(double x)
{
	m_cpt.y = x;
	emit paramChanged(m_cpt); 
}

void ParamDialog::recvThickness(double x)
{
	m_cpt.z = x;
	emit paramChanged(m_cpt); 
}

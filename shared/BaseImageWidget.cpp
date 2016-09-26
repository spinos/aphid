#include <QtGui>
#include "BaseImageWidget.h"

namespace aphid {

BaseImageWidget::BaseImageWidget(QWidget *parent)
    : QWidget(parent)
{
	m_margin.set(48, 48);
	m_translation.set(0.f, 0.f);
	m_scaling.set(1.f, 1.f);
}

void BaseImageWidget::paintEvent(QPaintEvent * /* event */)
{
    QPainter painter(this);
    painter.fillRect(rect(), backgroundCol() );

	painter.translate(m_translation.x, m_translation.y);
	painter.scale(m_scaling.x, m_scaling.y);	
	
	clientDraw(&painter);
}

QColor BaseImageWidget::backgroundCol() const
{ return Qt::white; }

void BaseImageWidget::resizeEvent(QResizeEvent * event)
{
	m_portSize = event->size();
	//qDebug()<<"BaseImageWidget resize to "<<m_portSize;
}

void BaseImageWidget::mousePressEvent(QMouseEvent *event)
{	
    m_lastMousePos = event->pos();
    if(event->modifiers() == Qt::AltModifier) 
        return;
    
    //processSelection(event);
}

void BaseImageWidget::mouseReleaseEvent(QMouseEvent *event)
{
    //processDeselection(event);
}

void BaseImageWidget::mouseMoveEvent(QMouseEvent *event)
{
    if(event->modifiers() == Qt::AltModifier)
        processCamera(event);
    //else 
      //  processMouseInput(event);

    m_lastMousePos = event->pos();
}

void BaseImageWidget::processCamera(QMouseEvent *event)
{
    int dx = event->x() - m_lastMousePos.x();
    int dy = event->y() - m_lastMousePos.y();
	
    if (event->buttons() & Qt::MidButton) {
		trackCamera(dx, dy);
    }
	else if (event->buttons() & Qt::RightButton) {
		zoomCamera(-dx / 2 + -dy / 2);
    }
	update();
}

void BaseImageWidget::trackCamera(int dx, int dy)
{
	m_translation.x += dx;
	m_translation.y += dy;
}

void BaseImageWidget::zoomCamera(int dx)
{
	const float rdx = 2.f * (float)dx/((float)m_portSize.width()+1);
	m_scaling.x += rdx;
	m_scaling.y += rdx;
	ClampInPlace<float>(m_scaling.x, 0.05f, 20.f);
	ClampInPlace<float>(m_scaling.y, 0.05f, 20.f);
}

void BaseImageWidget::clientDraw(QPainter * pr)
{}

const QSize & BaseImageWidget::portSize() const
{ return m_portSize; }

QSize BaseImageWidget::minimumSizeHint() const
{ return QSize(160, 160); }

bool BaseImageWidget::isLandscape() const
{ return m_portSize.width() > m_portSize.height(); }

void BaseImageWidget::setMargin(const int & h, const int & v)
{ m_margin.set(h, v); }

const Int2 & BaseImageWidget::margin() const
{ return m_margin; }

}
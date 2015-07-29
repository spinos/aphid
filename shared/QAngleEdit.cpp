/*
 *  QAngleEdit.cpp
 *  
 *
 *  Created by jian zhang on 7/30/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "QAngleEdit.h"

QAngleEdit::QAngleEdit(QWidget *parent)
	: QWidget(parent)
{
	m_value = 0.0;
	m_lowLimit = 0.0;
	m_highLimit = 3.14159269 * 2.0;
	setFocusPolicy(Qt::ClickFocus);
}

void QAngleEdit::setMin(double x)
{ 
	m_lowLimit = x; 
	update();
}

void QAngleEdit::setMax(double x)
{ 
	m_highLimit = x; 
	update();
}

void QAngleEdit::setValue(double x)
{
	m_value = x; 
	update();
}

QSize QAngleEdit::minimumSizeHint() const
{ return QSize(100, 100); }

QSize QAngleEdit::sizeHint() const
{ return QSize(100, 100); }


void QAngleEdit::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
	paintBackground(painter);
	paintDelta(painter);
	paintHandle(painter);
}

void QAngleEdit::paintBackground(QPainter & painter)
{
	QBrush br(Qt::lightGray);
	QPen pn(Qt::NoPen);
	painter.setPen(pn);
	painter.setBrush(br);
	painter.translate(50,50);
	QRectF square(-40,-40,80,80);
	painter.rotate(toDeg(-m_lowLimit));
	painter.drawPie(square, 0, toDeg(m_highLimit - m_lowLimit) * 16);	
}

void QAngleEdit::paintDelta(QPainter & painter)
{
	if(!m_active) return;
	double low = m_last;
	double d = m_value - m_last;
	if(d<0.0) {
		d = -d;
		low = m_value;
	}
	if(d<1e-3) return;
	
	painter.resetTransform();
	painter.translate(50,50);
	QRectF square(-40,-40,80,80);
	QBrush br(Qt::gray);
	QPen pn(Qt::NoPen);
	painter.setPen(pn);
	painter.setBrush(br);
	painter.rotate(toDeg(-m_lowLimit));
	
	
	painter.drawPie(square, toDeg(low - m_lowLimit) * 16, toDeg(d) * 16);
}

void QAngleEdit::paintHandle(QPainter & painter)
{
	painter.resetTransform();
	QPen pn(Qt::black);
	painter.setPen(pn);
	painter.setBrush(Qt::white);
	painter.drawLine(QPointF(50,50), toDrawSpace(m_value));
	QRectF box(-4,-4,8,8);
	painter.resetTransform();
	painter.translate(toDrawSpace(m_value));
	painter.drawEllipse(box);
}

QPointF QAngleEdit::toDrawSpace(double x) const
{ return QPointF(50 + cos(x) * 40, 50 - sin(x) * 40); }

double QAngleEdit::toValueSpace(double x, double y, bool & status) const
{
	QPointF p(x, 100 - y);
	p -= QPointF(50, 50);
	double l = sqrt(p.x()*p.x()+p.y()*p.y());
	if(l<1e-2) {
		status = false;
		return 0;
	}
	
	p /= l;
	double a = acos(p.x());
	if(p.y() < 0.0) {
		if(m_lowLimit<0.0) a *= -1.0;
		else a = 3.14159269 * 2.0 - a;
	}

	if(a > m_lowLimit && a < m_highLimit) status = true;
	else status = false;
	
	return a;
}

void QAngleEdit::mousePressEvent(QMouseEvent *event)
{
	m_active = false;
	QPointF pmouse(event->x(), event->y());
	QPointF tohandle = toDrawSpace(m_value) - pmouse;
	if(tohandle.manhattanLength() < 13.0) {
		m_last = m_value;
		m_active = true;
	}
}

void QAngleEdit::mouseMoveEvent(QMouseEvent *event)
{
	if(!m_active) return;
	bool stat;
	double val = toValueSpace(event->x(), event->y(), stat);
	if(!stat) return;
	
	m_value = val;
	emit valueChanged(m_value);
	update();
}

void QAngleEdit::mouseReleaseEvent(QMouseEvent *event)
{ 
	m_active = false; 
	update();
}

double QAngleEdit::toDeg(double a) const
{ return a * 180.0 / 3.14159269; }

double QAngleEdit::value() const
{ return m_value; }
//:~
/*
 *  QSplineEdit.cpp
 *  
 *
 *  Created by jian zhang on 7/28/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "QSplineEdit.h"

QSplineEdit::QSplineEdit(QWidget *parent)
	: QWidget(parent)
{
// default state
	m_startValue = 1.;
	m_endValue = .5;
	m_startCvx = .5; 
	m_startCvy = 1.;
	m_endCvx = .5; 
	m_endCvy = .5;
	setFocusPolicy(Qt::ClickFocus);
}

QSize QSplineEdit::minimumSizeHint() const
{
	return QSize(200, 100);
}

QSize QSplineEdit::sizeHint() const
{
    return QSize(200, 100);
}
	
void QSplineEdit::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
	
	paintBackground(painter);
	paintSpline(painter);
	paintControlLines(painter);
	paintControlHandles(painter);
}

QPointF QSplineEdit::toDrawSpace(double x, double y) const
{ return QPointF(10 + x * 180, 90 - y * 80); }

QPointF QSplineEdit::toValueSpace(double x, double y) const
{ return QPointF(toValueSpaceX(x), toValueSpaceY(y)); }

double QSplineEdit::toValueSpaceX(double x) const
{
	if(x > 190) x = 190;
	if(x < 10) x = 10;
	return (x - 10) / 180;
}

double QSplineEdit::toValueSpaceY(double y) const
{
	if(y > 90) y = 90;
	if(y < 10) y = 10;
	y = (y - 10) / 80;
	return 1.0 - y;
}

void QSplineEdit::paintBackground(QPainter & pnt)
{
	QBrush brbg(Qt::darkGray);
	pnt.setBackground(brbg);
	
	QRectF square(0,0,200,100);
	pnt.fillRect(square, brbg);
}

void QSplineEdit::paintSpline(QPainter & pnt)
{
	QBrush br(Qt::NoBrush);
	QPen pn(Qt::black, 2.0);
	pnt.setPen(pn);
	pnt.setBrush(br);
	QPainterPath bezierPath;
    bezierPath.moveTo(toDrawSpace(0, m_startValue));
    bezierPath.cubicTo(toDrawSpace(m_startCvx, m_startCvy), 
						toDrawSpace(m_endCvx, m_endCvy),
						toDrawSpace(1, m_endValue));
	
    pnt.drawPath(bezierPath);
}

void QSplineEdit::paintControlLines(QPainter & pnt)
{
	QBrush br(Qt::NoBrush);
	QPen pn(Qt::black);
	pnt.setPen(pn);
	pnt.setBrush(br);
	
	QPainterPath startControl;
	startControl.moveTo(toDrawSpace(0, m_startValue));
	startControl.lineTo(toDrawSpace(m_startCvx, m_startCvy));
	pnt.drawPath(startControl);
	
	QPainterPath endControl;
	endControl.moveTo(toDrawSpace(1, m_endValue));
	endControl.lineTo(toDrawSpace(m_endCvx, m_endCvy));
	pnt.drawPath(endControl);
}

void QSplineEdit::paintControlHandles(QPainter & pnt)
{
	QBrush br(Qt::lightGray);
	QPen pn(Qt::black);
	pnt.setPen(pn);
	pnt.setBrush(br);
	
	QRectF square(-4,-4,8,8);
	pnt.translate(toDrawSpace(0, m_startValue));
	pnt.drawRect(square);
	pnt.resetTransform();
	pnt.translate(toDrawSpace(1, m_endValue));
	pnt.drawRect(square);
	
	pnt.resetTransform();
	pnt.translate(toDrawSpace(m_startCvx, m_startCvy));
	pnt.drawEllipse(square);
	
	pnt.resetTransform();
	pnt.translate(toDrawSpace(m_endCvx, m_endCvy));
	pnt.drawEllipse(square);
}

void QSplineEdit::mousePressEvent(QMouseEvent *event)
{
	selectControlHandle(event->x(), event->y());
}

void QSplineEdit::mouseMoveEvent(QMouseEvent *event)
{
	moveControlHandle(event->x(), event->y());
}

void QSplineEdit::selectControlHandle(int x, int y)
{
	m_selected = HNone;
	
	QPointF pmouse(x, y);
	
	QPointF tostart = toDrawSpace(0, m_startValue) - pmouse;
	if(tostart.manhattanLength() < 5.0) {
		m_selected = HStart;
		return;
	}
	
	QPointF toend = toDrawSpace(1, m_endValue) - pmouse;
	if(toend.manhattanLength() < 5.0) {
		m_selected = HEnd;
		return;
	}
	
	QPointF toone = toDrawSpace(m_startCvx, m_startCvy) - pmouse;
	if(toone.manhattanLength() < 5.0) {
		m_selected = HControlLeft;
		return;
	}
	
	QPointF totwo = toDrawSpace(m_endCvx, m_endCvy) - pmouse;
	if(totwo.manhattanLength() < 5.0) {
		m_selected = HControlRight;
		return;
	}
}

void QSplineEdit::moveControlHandle(int x, int y)
{
	if(m_selected == HNone) return;
	
	switch (m_selected) {
		case HStart:
			moveStart(y);
			break;
		case HEnd:
			moveEnd(y);
			break;
		case HControlLeft:
			moveControlLeft(x, y);
			break;
		case HControlRight:
			moveControlRight(x, y);
			break;
		default:
			break;
	}
	update();
}

void QSplineEdit::moveStart(int y)
{ 
	const double v0 = m_startValue;
	m_startValue = toValueSpaceY(y); 
	const double dv = m_startValue - v0;
	m_startCvy += dv;
	
	emit valueChanged(QPointF(m_startValue, m_endValue));
	emit leftControlChanged(QPointF(m_startCvx, m_startCvy));
}

void QSplineEdit::moveEnd(int y)
{ 
	const double v0 = m_endValue;
	m_endValue = toValueSpaceY(y); 
	const double dv = m_endValue - v0;
	m_endCvy += dv;
	
	emit valueChanged(QPointF(m_startValue, m_endValue));
	emit rightControlChanged(QPointF(m_endCvx, m_endCvy));
}

void QSplineEdit::moveControlLeft(int x, int y)
{
	m_startCvx = toValueSpaceX(x);
	m_startCvy = toValueSpaceY(y);
	emit leftControlChanged(QPointF(m_startCvx, m_startCvy));
}

void QSplineEdit::moveControlRight(int x, int y)
{
	m_endCvx = toValueSpaceX(x);
	m_endCvy = toValueSpaceY(y);
	emit rightControlChanged(QPointF(m_endCvx, m_endCvy));
}
//:~
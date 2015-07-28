/*
 *  QSplineEdit.h
 *  
 *
 *  Created by jian zhang on 7/28/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef Q_SPLINE_EDIT_H
#define Q_SPLINE_EDIT_H

#include <QWidget>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class QSplineEdit : public QWidget 
{
	Q_OBJECT
public:
	QSplineEdit(QWidget *parent = 0);
	
	QSize minimumSizeHint() const;
    QSize sizeHint() const;
private slots:
	
protected:
	void paintEvent(QPaintEvent *event);
	virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
	
signals:
	void valueChanged(QPointF p);
	void leftControlChanged(QPointF p);
	void rightControlChanged(QPointF p);
	
private:
	QPointF toDrawSpace(double x, double y) const;
	QPointF toValueSpace(double x, double y) const;
	double toValueSpaceX(double x) const;
	double toValueSpaceY(double y) const;
	void paintBackground(QPainter & pnt);
	void paintSpline(QPainter & pnt);
	void paintControlLines(QPainter & pnt);
	void paintControlHandles(QPainter & pnt);
	void selectControlHandle(int x, int y);
	void moveControlHandle(int x, int y);
	void moveStart(int y);
	void moveEnd(int y);
	void moveControlLeft(int x, int y);
	void moveControlRight(int x, int y);
private:
	QLabel * m_label;
	
	double m_startValue, m_endValue;
	double m_startCvx, m_startCvy;
	double m_endCvx, m_endCvy;
	
	enum SelectedHandle {
		HNone = 0,
		HStart = 1,
		HEnd = 2,
		HControlLeft = 3,
		HControlRight = 4
	};
	
	SelectedHandle m_selected;
};
#endif
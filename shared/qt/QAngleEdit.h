/*
 *  QAngleEdit.h
 *  
 *
 *  Created by jian zhang on 7/30/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef Q_ANGLE_EDIT_H
#define Q_ANGLE_EDIT_H

#include <QWidget>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class QAngleEdit : public QWidget 
{
	Q_OBJECT
public:
	QAngleEdit(const QString & name, QWidget *parent = 0);
	
	void setMin(double x);
	void setMax(double x);
	void setValue(double x);
	
	QSize minimumSizeHint() const;
    QSize sizeHint() const;
	
	double value() const;
private slots:
	
protected:
	void paintEvent(QPaintEvent *event);
	virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
	
signals:
	void valueChanged(double a);
	
private:
	void paintBackground(QPainter & painter);
	void paintDelta(QPainter & painter);
	void paintHandle(QPainter & painter);
	QPointF toDrawSpace(double x) const;
	double toValueSpace(double x, double y, bool & status) const;
	double toDeg(double a) const;
private:
	QString m_name; 
	double m_value, m_lowLimit, m_highLimit, m_last;
	bool m_active;
};

#endif
/*
 *  2D view with track and zoom
 *
 */
 
#ifndef APHID_IMAGE_WIDGET_H
#define APHID_IMAGE_WIDGET_H
#include <QPixmap>
#include <QImage>
#include <QWidget>
#include "Vector2F.h"

namespace aphid {

class BaseImageWidget : public QWidget
{
    Q_OBJECT

public:
    BaseImageWidget(QWidget *parent = 0);

	void paintEvent(QPaintEvent *event);
	void resizeEvent(QResizeEvent *event);
	void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
	virtual QSize	minimumSizeHint() const;
	
protected:
	virtual void clientDraw(QPainter * pr);
	virtual QColor backgroundCol() const;
	
	const QSize & portSize() const; 
	
private:
	void processCamera(QMouseEvent *event);
	void trackCamera(int dx, int dy);
	void zoomCamera(int dx);
	
private:
	QSize m_portSize;
	QPoint m_lastMousePos;
	Vector2F m_translation, m_scaling;
	
};

}
#endif
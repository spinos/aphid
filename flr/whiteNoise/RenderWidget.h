
#ifndef MANDELBROTWIDGET_H
#define MANDELBROTWIDGET_H

#include <QPixmap>
#include <QWidget>

class RenderThread;
class RenderInterface;

class RenderWidget : public QWidget
{
    Q_OBJECT

public:
    RenderWidget(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

private slots:
    void updatePixmap();
	
private:
    void zoom(double zoomFactor);
    void scroll(int deltaX, int deltaY);

	RenderInterface* m_interface;
    RenderThread* thread;
	
    QPoint pixmapOffset;
    QPoint lastDragPos;
    double centerX;
    double centerY;
    double pixmapScale;
    double curScale;
};
//! [0]

#endif

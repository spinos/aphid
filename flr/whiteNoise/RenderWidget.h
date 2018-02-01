
#ifndef MANDELBROTWIDGET_H
#define MANDELBROTWIDGET_H

#include <QPixmap>
#include <QWidget>

namespace aphid {
class PerspectiveCamera;
}

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
	void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
	
private slots:
    void updatePixmap();
	
private:
	void processCamera(QMouseEvent *event);

	aphid::PerspectiveCamera* m_perspCamera;
	RenderInterface* m_interface;
    RenderThread* thread;
	
    QPoint pixmapOffset;
    QPoint m_lastMousePos;
    int m_dx, m_dy;
	
};

#endif

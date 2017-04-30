
#include <QPixmap>
#include <QWidget>

#include "renderthread.h"

//! [0]
class MandelbrotWidget : public QWidget
{
    Q_OBJECT

public:
    MandelbrotWidget(aphid::CudaRender * r, 
					QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void keyPressEvent(QKeyEvent *event);
	void focusInEvent(QFocusEvent * event);
	void focusOutEvent(QFocusEvent * event);
	
private slots:
    void updatePixmap(const QImage &image);
	
private:
	void processCamera(QMouseEvent *event);
	
private:
    RenderThread thread;
    QPixmap pixmap;
	QPoint m_lastMousePos;
};


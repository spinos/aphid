
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

private slots:
    void updatePixmap(const QImage &image);
	
private:
    RenderThread thread;
    QPixmap pixmap;

};


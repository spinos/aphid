#include <QTime>
#include <QPixmap>
#include <QWidget>

#include "lbmSolver.h"

//! [0]
class MandelbrotWidget : public QWidget
{
    Q_OBJECT

public:
    MandelbrotWidget(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);

private slots:
    void updatePixmap(const QImage &image, const unsigned &step);
	void simulate(); 

private:
    RenderThread thread;
    QPixmap pixmap;
	float _scaleFactor;
	unsigned _step;
	QTime _record_time;
};


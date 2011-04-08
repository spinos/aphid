#include <QtGui>
#include <math.h>
#include <sstream>
#include "lbmWidget.h"

MandelbrotWidget::MandelbrotWidget(QWidget *parent)
    : QWidget(parent)
{

    qRegisterMetaType<QImage>("QImage");
    connect(&thread, SIGNAL(renderedImage(QImage, unsigned)),
            this, SLOT(updatePixmap(QImage, unsigned)));

    setWindowTitle(tr("LBM 2D Fluid 128 X 128"));

    resize(512, 512);
	
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(41);
	
	_record_time.start();
	
	impulsePos = QPoint(0,0);
}

void MandelbrotWidget::paintEvent(QPaintEvent * /* event */)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);
	painter.setPen(Qt::white);
    if (pixmap.isNull()) {
        painter.drawText(rect(), Qt::AlignCenter,
                         tr("Rendering initial image, please wait..."));
        return;
    }
	painter.scale(_scaleFactor, _scaleFactor);
	painter.drawPixmap(QPoint(), pixmap);
	
	painter.scale(1.f/_scaleFactor, 1.f/_scaleFactor);
	
	int frame_elapse_time = _record_time.elapsed();
	float fps = float(_step) /( frame_elapse_time / 1000.f);
	
	std::stringstream sst;
	sst.str("");
	sst<<"updates per second: "<<fps;
	painter.drawText(QPoint(0, 16), sst.str().c_str());
	sst.str("");
	sst<<"impulse: "<<impulsePos.x()<<" "<<impulsePos.y();
	painter.drawText(QPoint(0, 32), sst.str().c_str());
}

void MandelbrotWidget::resizeEvent(QResizeEvent * /* event */)
{
	QSize renderAreaSize = size();
	_scaleFactor = renderAreaSize.height() / 128.f;
	thread.render();
}

void MandelbrotWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
        impulsePos = event->pos();
}
//! [13]

//! [14]
void MandelbrotWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton) {
        QPoint disp = event->pos() - impulsePos;
		float dx = float(disp.x());
		float dy = float(disp.y());
		float vscaling = sqrt(dx*dx + dy*dy);
		if(vscaling < 0.1f) vscaling = 0.1f;
		dx /= vscaling;
		dy /= vscaling;
		
		float fx = (float)impulsePos.x()/size().width();
		float fy = (float)impulsePos.y()/size().height();
		impulsePos = event->pos();
		thread.addImpulse(fx*128, fy*128, dx, dy);
    }
}
//! [14]

//! [15]
void MandelbrotWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        impulsePos = QPoint(0,0);
		
    }
}

void MandelbrotWidget::updatePixmap(const QImage &image, const unsigned &step)
{

    pixmap = QPixmap::fromImage(image);
	_step = step;
    update();
}

void MandelbrotWidget::simulate()
{
    update();

	thread.render();
}
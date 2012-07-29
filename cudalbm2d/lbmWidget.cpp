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

    setWindowTitle(tr("LBM 2D Fluid 192 X 120"));

    resize(thread.solverWidth() * 4, thread.solverHeight() * 4);
	
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(37);
	
	_record_time.start();
	_num_update = 0;
	
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
	float fps = float(_num_update) /( frame_elapse_time / 1000.f);
	
	std::stringstream sst;
	sst.str("");
	sst<<"updates per second: "<<fps;
	painter.drawText(QPoint(0, 16), sst.str().c_str());
}

void MandelbrotWidget::resizeEvent(QResizeEvent * /* event */)
{
	QSize renderAreaSize = size();
	_scaleFactor = renderAreaSize.height() / thread.solverHeight();
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
		if(vscaling < 2.f) 
		{
			dx = dy = 0.f;
		}
		if(vscaling > 20.f) 
		{
			vscaling = vscaling / 20.f;
			dx /= vscaling;
			dy /= vscaling;
		}
		
		dx /= 40.f;
		dy /= 40.f;
		
		float fx = (float)impulsePos.x()/size().width();
		if(fx < 0.f) fx = 0.f;
		if(fx > 1.f) fx = 1.f;
		float fy = (float)impulsePos.y()/size().height();
		if(fy < 0.f) fy = 0.f;
		if(fy > 1.f) fy = 1.f;
		
		impulsePos = event->pos();
		thread.addImpulse(fx*(thread.solverWidth() - 1), fy*(thread.solverHeight()-1), dx, dy);
		
    }
	else if(event->buttons() & Qt::RightButton) {
		QPoint hit = event->pos();
		float fx = (float)hit.x()/size().width();
		if(fx < 0.f) fx = 0.f;
		if(fx > 1.f) fx = 1.f;
		float fy = (float)hit.y()/size().height();
		if(fy < 0.f) fy = 0.f;
		if(fy > 1.f) fy = 1.f;
		thread.addObstacle(fx*(thread.solverWidth() - 1), fy*(thread.solverHeight()-1));
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
	_num_update++;
	update();	
}

void MandelbrotWidget::simulate()
{
    update();

	thread.render();
}
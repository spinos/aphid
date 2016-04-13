#include <QtGui>

#include "whitenoisewidget.h"

MandelbrotWidget::MandelbrotWidget(aphid::CudaRender * r, 
									QWidget *parent)
    : QWidget(parent)
{
	thread.setR(r);
	
	qRegisterMetaType<QImage>("QImage");
    connect(&thread, SIGNAL(renderedImage(QImage)),
            this, SLOT(updatePixmap(QImage)));

	//QTimer *timer = new QTimer(this);
	//connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	//timer->start(40);
}

void MandelbrotWidget::paintEvent(QPaintEvent * /* event */)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (pixmap.isNull()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter,
                         tr("Rendering initial image, please wait..."));
        return;
    }

	painter.drawPixmap(QPoint(), pixmap);
}

void MandelbrotWidget::resizeEvent(QResizeEvent * /* event */)
{
	thread.render(size());
}

void MandelbrotWidget::mousePressEvent(QMouseEvent *event)
{	
    m_lastMousePos = event->pos();
    //if(event->modifiers() == Qt::AltModifier) 
      //  return;
    
    //processSelection(event);
}

void MandelbrotWidget::mouseMoveEvent(QMouseEvent *event)
{
    if(event->modifiers() == Qt::AltModifier)
		processCamera(event);
    //    processCamera(event);
    else {}
      //  processMouseInput(event);

    m_lastMousePos = event->pos();
}

void MandelbrotWidget::mouseReleaseEvent(QMouseEvent *event)
{}

void MandelbrotWidget::updatePixmap(const QImage &image)
{

    pixmap = QPixmap::fromImage(image);
    update();
}

void MandelbrotWidget::processCamera(QMouseEvent *event)
{
    int dx = event->x() - m_lastMousePos.x();
    int dy = event->y() - m_lastMousePos.y();
    if (event->buttons() & Qt::LeftButton) {
        thread.tumble(dx, dy);
    } 
	else if (event->buttons() & Qt::MidButton) {
		thread.track(dx, dy);
    }
	else if (event->buttons() & Qt::RightButton) {
		thread.zoom(-dx / 2 - dy / 2 );
    }
}

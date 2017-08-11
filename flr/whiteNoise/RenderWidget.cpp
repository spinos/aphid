
#include <QtGui>

#include <math.h>

#include "RenderWidget.h"
#include <interface/RenderThread.h>
#include <interface/RenderInterface.h>

const double DefaultCenterX = -0.637011f;
const double DefaultCenterY = -0.0395159f;
const double DefaultScale = 0.00403897f;

const double ZoomInFactor = 0.8f;
const double ZoomOutFactor = 1 / ZoomInFactor;
const int ScrollStep = 20;

RenderWidget::RenderWidget(QWidget *parent)
    : QWidget(parent)
{
    centerX = DefaultCenterX;
    centerY = DefaultCenterY;
    pixmapScale = DefaultScale;
    curScale = DefaultScale;

	m_interface = new RenderInterface;
	m_interface->createImage(1024, 800);
	thread = new RenderThread(m_interface);
	
    connect(thread, SIGNAL(renderedImage()),
            this, SLOT(updatePixmap()));

    resize(550, 400);
}

void RenderWidget::paintEvent(QPaintEvent * /* event */)
{
//qDebug()<<" paint event";
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);
	
	painter.drawImage(pixmapOffset, m_interface->getQImage() );

}

void RenderWidget::resizeEvent(QResizeEvent * /* event */)
{
	thread->render(centerX, centerY, curScale, size());
}

void RenderWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_Plus:
        zoom(ZoomInFactor);
        break;
    case Qt::Key_Minus:
        zoom(ZoomOutFactor);
        break;
    case Qt::Key_Left:
        scroll(-ScrollStep, 0);
        break;
    case Qt::Key_Right:
        scroll(+ScrollStep, 0);
        break;
    case Qt::Key_Down:
        scroll(0, -ScrollStep);
        break;
    case Qt::Key_Up:
        scroll(0, +ScrollStep);
        break;
    default:
        QWidget::keyPressEvent(event);
    }
}
//! [11]

//! [12]
void RenderWidget::wheelEvent(QWheelEvent *event)
{
    int numDegrees = event->delta() / 8;
    double numSteps = numDegrees / 15.0f;
    zoom(pow(ZoomInFactor, numSteps));
}
//! [12]

//! [13]
void RenderWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
        lastDragPos = event->pos();
}
//! [13]

//! [14]
void RenderWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton) {
        pixmapOffset += event->pos() - lastDragPos;
        lastDragPos = event->pos();
        update();
    }
}
//! [14]

//! [15]
void RenderWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        pixmapOffset += event->pos() - lastDragPos;
        lastDragPos = QPoint();

        int deltaX = (width() - m_interface->xres()) / 2 - pixmapOffset.x();
        int deltaY = (height() - m_interface->yres()) / 2 - pixmapOffset.y();
        scroll(deltaX, deltaY);
    }
}
//! [15]

//! [16]
void RenderWidget::updatePixmap()
{
    if (!lastDragPos.isNull())
        return;

    pixmapOffset = QPoint();
    lastDragPos = QPoint();
    pixmapScale = 1.0;
    update();
}

void RenderWidget::zoom(double zoomFactor)
{
    curScale *= zoomFactor;
    update();
    thread->render(centerX, centerY, curScale, size());
}

void RenderWidget::scroll(int deltaX, int deltaY)
{
    centerX += deltaX * curScale;
    centerY += deltaY * curScale;
    update();
    thread->render(centerX, centerY, curScale, size());
}

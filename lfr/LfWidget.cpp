#include <QtGui>
#include "LfWidget.h"
#include "LfWorld.h"

namespace lfr {

LfWidget::LfWidget(LfWorld * world, QWidget *parent)
    : QWidget(parent)
{
	m_world = world;

    // setWindowTitle(tr("Image Atoms"));

    // resize(500, 400);
	
	
	// QTimer *timer = new QTimer(this);
	// connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	// timer->start(40);
}

void LfWidget::paintEvent(QPaintEvent * /* event */)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (m_pixmap.isNull()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter,
                         tr("Rendering initial image, please wait..."));
        return;
    }

	// int x, y;
	// m_world->param()->getDictionaryImageSize(x, y);

	painter.scale(2,2);	
	painter.drawPixmap(QPoint(2,2), m_pixmap);
	
}

void LfWidget::resizeEvent(QResizeEvent * /* event */)
{
    // m_thread->render(size());
}

void LfWidget::recvDictionary(const QImage &image)
{
	m_pixmap = QPixmap::fromImage(image);
	update();
}

}
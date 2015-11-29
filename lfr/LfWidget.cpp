#include <QtGui>
#include "LfWidget.h"
#include "LfThread.h"
#include "LfWorld.h"

namespace lfr {

LfWidget::LfWidget(LfWorld * world, QWidget *parent)
    : QWidget(parent)
{
	m_world = world;
	m_thread = new LfThread(world, this);
    qRegisterMetaType<QImage>("QImage");
    connect(m_thread, SIGNAL(sendInitialDictionary(QImage)),
            this, SLOT(updatePixmap(QImage)));
			
	connect(m_thread, SIGNAL(sendDictionary(QImage)),
            this, SLOT(recvDictionary(QImage)));
			
	connect(m_thread, SIGNAL(sendSparsity(QImage)),
            this, SLOT(recvSparsity(QImage)));

    setWindowTitle(tr("Image Atoms"));

    resize(500, 400);
	
	m_thread->initAtoms();
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

	painter.drawPixmap(QPoint(2,2), m_pixmap);
	
	if(m_sparsityPix.isNull()) return;
	
	int x, y;
	m_world->param()->getDictionaryImageSize(x, y);
	painter.drawPixmap(QPoint(x+4, 2), m_sparsityPix);
}

void LfWidget::resizeEvent(QResizeEvent * /* event */)
{
    // m_thread->render(size());
}

void LfWidget::updatePixmap(const QImage &image)
{
    m_pixmap = QPixmap::fromImage(image);
    update();
	m_thread->beginLearn();
}

void LfWidget::recvDictionary(const QImage &image)
{
	m_pixmap = QPixmap::fromImage(image);
    update();
}

void LfWidget::recvSparsity(const QImage &image)
{
	m_sparsityPix = QPixmap::fromImage(image);
	update();
}

void LfWidget::simulate()
{
    update();
    //m_thread->render(size());
}

}
#include <QtGui>
#include "StatisticWidget.h"
#include "LfWorld.h"

namespace lfr {

StatisticWidget::StatisticWidget(LfWorld * world, QWidget *parent)
    : QWidget(parent)
{
	int x, y;
	world->param()->getDictionaryImageSize(x, y);
	x *= 2;
	y *= 2;

    setWindowTitle(tr("Statistics"));

    resize(300, 200);
}

void StatisticWidget::paintEvent(QPaintEvent * /* event */)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

	if(m_sparsityPix.isNull()) return;
	
	painter.drawPixmap(QPoint(2, 2), m_sparsityPix);
}

void StatisticWidget::resizeEvent(QResizeEvent * /* event */)
{
}

void StatisticWidget::recvSparsity(const QImage &image)
{
	m_sparsityPix = QPixmap::fromImage(image);
	update();
}

}
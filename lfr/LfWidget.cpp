#include <QtGui>
#include "LfWidget.h"

LfWidget::LfWidget(QWidget *parent)
    : QWidget(parent)
{

    qRegisterMetaType<QImage>("QImage");
    connect(&thread, SIGNAL(renderedImage(QImage)),
            this, SLOT(updatePixmap(QImage)));

    setWindowTitle(tr("White Noise"));

    resize(500, 400);
	
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(40);
}

void LfWidget::paintEvent(QPaintEvent * /* event */)
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

void LfWidget::resizeEvent(QResizeEvent * /* event */)
{
    thread.render(size());
}

void LfWidget::updatePixmap(const QImage &image)
{
    pixmap = QPixmap::fromImage(image);
    update();
}

void LfWidget::simulate()
{
    update();
    thread.render(size());
}
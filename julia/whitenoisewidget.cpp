#include <QtGui>

#include <math.h>

#include "whitenoisewidget.h"

MandelbrotWidget::MandelbrotWidget(QWidget *parent)
    : QWidget(parent)
{

    qRegisterMetaType<QImage>("QImage");
    connect(&thread, SIGNAL(renderedImage(QImage)),
            this, SLOT(updatePixmap(QImage)));

    setWindowTitle(tr("White Noise"));

    resize(540, 480);
	
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

void MandelbrotWidget::updatePixmap(const QImage &image)
{

    pixmap = QPixmap::fromImage(image);
    update();
}

#ifndef LFWIDGET_H
#define LFWIDGET_H
#include <QPixmap>
#include <QWidget>

namespace lfr {

class LfWorld;

class LfWidget : public QWidget
{
    Q_OBJECT

public:
    LfWidget(LfWorld * world, QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);

private slots:
    void recvDictionary(const QImage &image);

private:
	LfWorld * m_world;
    QPixmap m_pixmap;
};

}
#endif
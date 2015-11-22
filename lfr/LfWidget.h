#include <QPixmap>
#include <QWidget>
#include "LfThread.h"

class LfWidget : public QWidget
{
    Q_OBJECT

public:
    LfWidget(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);

private slots:
    void updatePixmap(const QImage &image);
	void simulate(); 

private:
    LfThread thread;
    QPixmap pixmap;

};


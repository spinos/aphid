#include <QPixmap>
#include <QWidget>

namespace lfr {

class LfWorld;
class StatisticWidget : public QWidget
{
    Q_OBJECT

public:
    StatisticWidget(LfWorld * world, QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);

private slots:
   void recvSparsity(const QImage &image);
	
private:
	QPixmap m_sparsityPix;
};

}
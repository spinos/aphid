#ifndef STATISTICDIALOG_H
#define STATISTICDIALOG_H
#include <QPixmap>
#include <QDialog>
#include <deque>

namespace lfr {

class LfWorld;
class StatisticDialog : public QDialog
{
    Q_OBJECT

public:
    StatisticDialog(LfWorld * world, QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);

public slots:
   void recvSparsity(const QImage &image);
   void recvPSNR(float ratio);
	
private:
	QPixmap m_sparsityPix;
	std::deque<int> m_psnrs;
};

}
#endif
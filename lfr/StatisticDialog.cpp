#include <QtGui>
#include "StatisticDialog.h"
#include "LfWorld.h"

namespace lfr {

StatisticDialog::StatisticDialog(LfWorld * world, QWidget *parent)
    : QDialog(parent)
{
	int x, y;
	world->param()->getDictionaryImageSize(x, y);
	x *= 2;
	y *= 2;

    setWindowTitle(tr("Statistics"));

    resize(384, 256);
}

void StatisticDialog::paintEvent(QPaintEvent * /* event */)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

	if(m_sparsityPix.isNull()) return;
	
	painter.translate(0, 128);	
	painter.rotate(-90);
	painter.scale(1,2);
	painter.drawPixmap(QPoint(), m_sparsityPix);
	
	if(m_psnrs.size() < 1) return;
	painter.resetTransform();
	painter.translate(0, 240);
	QPen pn(Qt::green);
	painter.setPen(pn);
	painter.setBrush(Qt::white);
	
	if(m_psnrs.size() > 1) {
		int i = 1;
		for(;i<m_psnrs.size();i++) {
			painter.drawLine(QPoint( (i-1)*2, m_psnrs[i-1] ), QPoint( i*2, m_psnrs[i] ) );
		}
	}
	else
		painter.drawPoint ( 0, m_psnrs[0] );	
}

void StatisticDialog::resizeEvent(QResizeEvent * /* event */)
{
}

void StatisticDialog::recvSparsity(const QImage &image)
{
	m_sparsityPix = QPixmap::fromImage(image);
	update();
}

void StatisticDialog::recvPSNR(float ratio)
{
	if(m_psnrs.size() >= 256) 
		m_psnrs.pop_front();
	
	int iratio = ratio * 2;
/// upper limit 55
	if(iratio > 110) iratio = 110;
/// lower limit -10
	if(iratio < -10) iratio = -10;
	m_psnrs.push_back(-iratio);
}

}
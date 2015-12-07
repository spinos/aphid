#include <QtGui>
#include "StatisticDialog.h"
#include "LfMachine.h"

namespace lfr {

StatisticDialog::StatisticDialog(LfMachine * world, QWidget *parent)
    : QDialog(parent)
{
	int x, y;
	world->param()->getDictionaryImageSize(x, y);
	x *= 2;
	y *= 2;

    setWindowTitle(tr("Statistics"));

    resize(520, 240);
}

void StatisticDialog::paintEvent(QPaintEvent * /* event */)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::gray);
	QPen pn(Qt::black);
	pn.setWidth(1);
	painter.setPen(pn);
	painter.setBrush(Qt::NoBrush);
	
	drawSparsity(painter, 5, 45);
	drawPSNR(painter, 5, 240);
}

void StatisticDialog::drawSparsity(QPainter & painter, int baseX, int baseY)
{
	if(m_sparsityPix.isNull()) return;
	
	painter.translate(baseX + 26, baseY);	
	painter.rotate(-90);
	painter.scale(2, 1.75);
	painter.drawPixmap(QPoint(), m_sparsityPix);
	
	painter.resetTransform();
	
	const int tenY = baseY - 20;
	const int lft = baseX + 20;
	const int rgt = baseX + 480;
	
	painter.drawText( QPoint(baseX, tenY), tr("0.1"));
	painter.drawText( QPoint(rgt, tenY), tr("0.1"));
	painter.drawLine(QPoint(lft, tenY), QPoint(rgt, tenY));
	
	painter.drawText( QPoint(baseX, baseY), tr("0.0"));
	painter.drawText( QPoint(rgt, baseY), tr("0.0"));
	painter.drawLine(QPoint(lft, baseY), QPoint(rgt, baseY));
	
	painter.drawText( QPoint(baseX+200, baseY+20), tr("Sparsity"));
}

void StatisticDialog::drawPSNR(QPainter & painter, int baseX, int baseY)
{
	const int rgt = baseX + 480;
	const int lft = baseX + 20;
	const int twentY = baseY - 60;
	const int thirtY = baseY - 90;
	const int fortY = baseY - 120;
	
	painter.drawText( QPoint(baseX, twentY), tr("20"));
	painter.drawLine(QPoint(lft, twentY), QPoint(rgt, twentY));
	painter.drawText( QPoint(rgt, twentY), tr("20"));
	
	painter.drawText( QPoint(baseX, thirtY), tr("30"));
	painter.drawLine(QPoint(lft, thirtY), QPoint(rgt, thirtY));
	painter.drawText( QPoint(rgt, thirtY), tr("30"));
	
	painter.drawText( QPoint(baseX, fortY), tr("40"));
	painter.drawLine(QPoint(lft, fortY), QPoint(rgt, fortY));
	painter.drawText( QPoint(rgt, fortY), tr("40"));
	
	painter.drawText( QPoint(baseX+200, baseY - 30), tr("PSNR"));
	
	if(m_psnrs.size() < 1) return;
	
	QPen pn(Qt::blue);
	pn.setWidth(2);
	painter.setPen(pn);
	
	const int llft = baseX + 26;
	if(m_psnrs.size() > 1) {
		int i = 1;
		for(;i<m_psnrs.size();i++) {
			painter.drawLine(QPoint(llft + (i-1)*4, baseY+ m_psnrs[i-1] ), QPoint(llft + i*4, baseY+ m_psnrs[i] ) );
		}
	}
	else
		painter.drawPoint (llft, baseY+m_psnrs[0] );
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
	if(m_psnrs.size() >= 120) 
		m_psnrs.pop_front();
	
	int iratio = ratio * 3;
/// upper limit 55
	if(iratio > 165) iratio = 165;
/// lower limit -10
	if(iratio < -30) iratio = -30;
	m_psnrs.push_back(-iratio);
	update();
}

}
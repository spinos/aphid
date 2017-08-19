#include <QtGui>
#include "AboutGardenDlg.h"
#include <gar_common.h>

AboutGardenDlg::AboutGardenDlg(QWidget * parent) : QDialog(parent)
{
	QLabel* lab = new QLabel(tr("Garden"));
	lab->setFont(QFont("Helvetica", 18, 8));
	lab->setAlignment(Qt::AlignHCenter);
	QLabel* ver = new QLabel(tr("version %1.%2").arg(gar::DEV_VERSION_MAJOR ).arg(gar::DEV_VERSION_MINOR) );
	ver->setAlignment(Qt::AlignHCenter);
	QVBoxLayout *box = new QVBoxLayout;
    box->addWidget(lab);
	box->addWidget(ver);
	box->addStretch(1);
	setLayout(box);

    resize(200, 120);
}


#include "QFileNameEdit.h"

namespace aphid {

QFileNameEdit::QFileNameEdit(const QModelIndex & idx, QWidget * parent) : QModelEdit(idx, parent) {}

void QFileNameEdit::setValue(const std::string & x)
{
	m_fileName = x;
	QString t = m_fileName.c_str();
	setText(t);
}

std::string QFileNameEdit::value() 
{
	m_fileName = text().toUtf8().constData();
	return m_fileName;
}

std::string QFileNameEdit::pickFile()
{
    QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Choose image file"),
							tr("info"),
							tr("All Files (*);"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	return fileName.toUtf8().data();
}

}


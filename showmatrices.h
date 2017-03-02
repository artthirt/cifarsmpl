#ifndef SHOWMATRICES_H
#define SHOWMATRICES_H

#include <QWidget>
#include <QTimer>

#include "custom_types.h"

namespace Ui {
class ShowMatrices;
}

class ShowMatrices : public QWidget
{
	Q_OBJECT

public:
	explicit ShowMatrices(QWidget *parent = 0);
	~ShowMatrices();

	void setMat(const ct::Matf& mat, const ct::Size& sz, int K, int channels = 1);
	void saveMat(const QString& name, const ct::Matf& mat, const ct::Size& sz, int K, int channels = 1);

	void save2Image(const QString& name, int width, int height);

public slots:
	void onTimeout();

private:
	Ui::ShowMatrices *ui;

	ct::Matf m_mat;
	ct::Size m_sz;
	int m_channels;
	int m_K;

	QPoint m_pt;
	int m_offset;
	bool m_update;
	QTimer m_timer;

	// QWidget interface
protected:
	virtual void paintEvent(QPaintEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mousePressEvent(QMouseEvent *event);

	void paint_cast(QPainter& painter, int width = -1, int height = -1, int offset = 0);
};

#endif // SHOWMATRICES_H

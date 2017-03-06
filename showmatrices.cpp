#include "showmatrices.h"
#include "ui_showmatrices.h"

#include "matops.h"
#include "imutils.h"

#include <QPainter>
#include <QMouseEvent>

ShowMatrices::ShowMatrices(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::ShowMatrices)
{
	ui->setupUi(this);

	setMouseTracking(true);
	m_update = false;
	m_offset = 0;

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(50);
}

ShowMatrices::~ShowMatrices()
{
	delete ui;
}

void ShowMatrices::setMat(const ct::Matf &mat, const ct::Size &sz, int K, int channels)
{
	m_sz = sz;
	m_K = K;
	m_channels = channels;
	mat.copyTo(m_mat);
	update();
}

void ShowMatrices::saveMat(const QString &name, const ct::Matf &mat, const ct::Size &sz, int K, int channels)
{
	m_sz = sz;
	m_K = K;
	m_channels = channels;
	mat.copyTo(m_mat);
	save2Image(name, 2048, 2048);
}

void ShowMatrices::save2Image(const QString &name, int width, int height)
{
	QImage im(width, height, QImage::Format_ARGB32);

	QPainter painter(&im);
	paint_cast(painter, width, height);

	im.save(name);
}

void ShowMatrices::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);

	paint_cast(painter, width(), height(), m_offset);
}

void ShowMatrices::paint_cast(QPainter &painter, int width, int height, int offset)
{
	if(m_mat.empty() || m_mat.cols != m_K)
		return;

	int x = 0, y = offset;

	int wd = 10;

	if(width == -1)
		width = this->width();
	if(height == -1)
		height = this->height();

	painter.fillRect(QRect(0, 0, width, height), Qt::white);

	QPen pen;
	pen.setWidth(2);

	for(int i = 0; i < m_K; ++i){
		pen.setWidth(2);

		ct::Matf m = m_mat.col(i);

		if(m.total() != m_sz.area() * m_channels)
			continue;

		float m1 = m.max();
		float m2 = m.min();

		int off_y = m_channels == 3? 1 : m_channels;

		if(m_channels == 1){
			m.set_dims(m_sz);
			painter.setPen(Qt::black);
			draw_W(painter, m, x, y, wd, m1, m2);
		}else{
			if(m_channels == 3){
				ct::Matf R, G, B;
				R = m.getRows(0 * m_sz.area(), m_sz.area());
				G = m.getRows(1 * m_sz.area(), m_sz.area());
				B = m.getRows(2 * m_sz.area(), m_sz.area());

				R.set_dims(m_sz);
				G.set_dims(m_sz);
				B.set_dims(m_sz);

				painter.setPen(Qt::black);
				draw_W(painter, R, G, B, x, y, wd, m1, m2);
			}else{
				int y_prev = y;
				for(int i = 0; i < m_channels; ++i){
					ct::Matf R;
					R = m.getRows(i * m_sz.area(), m_sz.area());
					R.set_dims(m_sz);
					painter.setPen(Qt::black);
					draw_W(painter, R, x, y, wd, m1, m2);

					pen.setColor(Qt::red);
					painter.setPen(pen);
					QPoint p1 = QPoint(x, y);
					QPoint p2 = p1 + QPoint(0,  wd * m_sz.height);
					painter.drawLine(p1, p2);

					pen.setColor(Qt::green);
					painter.setPen(pen);
					p1 = QPoint(x, y + wd * m_sz.height);
					p2 = QPoint(x + wd * m_sz.width, y + wd * m_sz.height);
					painter.drawLine(p1, p2);

					y += wd * m_sz.height;
				}
				y = y_prev;
			}
		}

		x += wd * m_sz.width;
		if(x + wd * m_sz.width > width){
			y += wd * m_sz.height * off_y;
			x = 0;

			pen.setWidth(3);
			pen.setColor(Qt::blue);
			painter.setPen(pen);
			painter.drawLine(0, y, width, y);
		}
	}
}

void ShowMatrices::onTimeout()
{
	if(m_update){
		m_update = false;
		update();
	}
}

void ShowMatrices::mousePressEvent(QMouseEvent *event)
{
	m_pt = event->pos();
	if(event->buttons().testFlag(Qt::RightButton)){
		m_offset = 0;
		m_update = true;
	}
}


void ShowMatrices::mouseMoveEvent(QMouseEvent *event)
{
	if(event->buttons().testFlag(Qt::LeftButton)){
		m_offset += event->pos().y() - m_pt.y();
		m_pt = event->pos();
		if(m_offset < -4000){
			m_offset = -4000;
		}
		if(m_offset > 200){
			m_offset = 200;
		}
		m_update = true;
	}
}

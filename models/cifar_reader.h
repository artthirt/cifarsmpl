#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <QFile>
#include <QVector>
#include <QByteArray>
#include <QTimer>

#include "custom_types.h"

struct TData{
	ushort labels;
	QByteArray data;
	QByteArray image;

	void toImage();
};

class cifar_reader : public QObject
{
	Q_OBJECT
public:
	enum{WidthIM = 32, HeightIM = 32, BppIM = 3};
	enum {CIFAR10 = 0, CIFAR100 = 1};

	cifar_reader();

	QVector<TData> &train(int batch, double percent = -1);
	QVector<QByteArray> &test();

	uint count();
	uint current_file();
	uint max_files() const;
	double current_percent() const;
//	ct::Matf& X();
//	ct::Matf& y();

public slots:
	void onTimeout();

private:
	uint m_data_source;
	uint m_count;
	uint m_current_file;
	uint m_current_batch;
	uint m_current_offset;
	double m_current_percent;

	QVector<TData> m_current_data;
	QFile m_current_object;
	QTimer m_timer;

//	ct::Matf m_X;
//	ct::Matf m_y;

	uint readCifar(const QString &fn, QVector< TData >& val, int batch = 100, int offset = -1);
	uint readCifar(QFile &file, QVector<TData> &val, int batch = 100, int offset = -1);

};

namespace ct{

/**
 * @brief image2mats
 * split image array to vector of matrix
 * @param image
 * @param w
 * @param h
 * @param bpp
 * @param mats
 */
template< typename T >
inline void image2mats(const QByteArray& image, int w, int h, int bpp, std::vector< ct::Mat_<T> >& mats)
{
	mats.resize(bpp);

	for(int i = 0; i < bpp; ++i){
		Mat_<T>& m = mats[i];
		m.setSize(h, w);

		T* dm = m.ptr();
		uchar* db = (uchar*)image.data() + i * w * h;
		for(int y = 0; y < h; ++y){
			for(int x = 0; x < w; ++x){
				dm[y * w + x] = db[y * w + x] / 255.;
			}
		}
	}
}

}

#endif // MNIST_READER_H

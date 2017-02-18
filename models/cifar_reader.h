#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <QFile>
#include <QVector>
#include <QByteArray>
#include <QTimer>
#include <vector>

#include "custom_types.h"

struct TData{
	ushort lb;			/// label
	QByteArray data;	/// raw data
	QByteArray image;	/// image data for output to QImage

	void toImage();
};

class cifar_reader : public QObject
{
	Q_OBJECT
public:
	enum{WidthIM = 32, HeightIM = 32, BppIM = 3};
	enum {CIFAR10 = 0, CIFAR100 = 1};

	cifar_reader();
	~cifar_reader();

	QVector<TData> &train(int batch, double percent);
	QVector<TData> &test(int beg, int count);
	uint count_test();

	void convToXy(const QVector< TData > &data, std::vector< ct::Matf >& X, ct::Matf *y = nullptr);

	bool getData(double percent, TData& data);
	void getTrain(int batch, std::vector< ct::Matf >& X, ct::Matf &y, std::vector<double> *percents = nullptr);
	bool getDataIt(double percent, int batch, QVector<TData> &data);
	void getTrainIt(double percent, int batch, std::vector< ct::Matf >& X, ct::Matf *y = nullptr);

	uint getTest(uint beg, uint batch, std::vector< ct::Matf >& X, ct::Matf &y);

	uint count();
	uint current_file();
	uint max_files() const;
	double current_percent() const;
//	ct::Matf& X();
//	ct::Matf& y();

	bool openDir(const QString& dir);
	QString currentDirectory() const;

	bool isBinDataExists()const ;

public slots:
	void onTimeout();
	void onTimeoutTest();

private:
	uint m_data_source;
	uint m_count;
	uint m_current_file;
	uint m_current_batch;
	uint m_current_offset;
	double m_current_percent;

	QVector<TData> m_current_data;
	QVector<TData> m_current_test;
	QFile m_current_object;
	QTimer m_timer;
	QTimer m_timer_test;

	QString m_directory;

	uint m_count_test;
	QFile m_current_test_object;

	void open_test_file();

//	ct::Matf m_X;
//	ct::Matf m_y;

	uint readCifar(const QString &fn, QVector< TData >& val, int batch = 100, int offset = -1);
	uint readCifar(QFile &file, QVector<TData> &val, int batch = 100, int offset = -1);

	uint readCifar1(const QString &fn, TData & val, int offset);
	uint readCifar1(QFile &file, TData &val, int offset);

	void loadSettings();
	void saveSettings();
};

namespace ct{

template< typename T >
inline void image2mat(uchar* db, int w, int h, int row, ct::Mat_<T>& mat)
{
	if(mat.empty())
		throw new std::invalid_argument("image2mat: empty mat");
	T* dm = mat.ptr();
	T* dmi = &dm[row * mat.cols];
	for(int y = 0; y < h; ++y){
		for(int x = 0; x < w; ++x){
			dmi[y * w + x] = db[y * w + x] / 255.;
		}
	}

}

template< typename T >
inline void image2mats(const QByteArray& image, int w, int h, int row,
					   ct::Mat_<T>& matR,
					   ct::Mat_<T>& matG,
					   ct::Mat_<T>& matB)
{
	uchar* dbR = (uchar*)image.data() + 0 * w * h;
	uchar* dbG = (uchar*)image.data() + 1 * w * h;
	uchar* dbB = (uchar*)image.data() + 2 * w * h;
	image2mat(dbR, w, h, row, matR);
	image2mat(dbG, w, h, row, matG);
	image2mat(dbB, w, h, row, matB);
}

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

#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;


int MNIST_reverse_int(int i)
{
	unsigned char ch1, ch2, ch3, ch4;

	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


void MNIST_read_images(string filename, vector<cv::Mat> &vec)
{
	vec.clear();
	ifstream file (filename.c_str(), ios::binary);

	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = MNIST_reverse_int(magic_number);

		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = MNIST_reverse_int(number_of_images);

		file.read((char*) &n_rows, sizeof(n_rows));
		n_rows = MNIST_reverse_int(n_rows);

		file.read((char*) &n_cols, sizeof(n_cols));
		n_cols = MNIST_reverse_int(n_cols);

		for(int i = 0; i < number_of_images; ++i)
		{
			cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_32F);

			for(int r = 0; r < n_rows; ++r)
			{
				for(int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*) &temp, sizeof(temp));
					float d = temp;
					tp.at<float>(r, c) = (d / 255.0) * 2.0 - 1.0;
				}
			}

			vec.push_back(tp);
		}
	}
}


void MNIST_read_labels(string filename, vector<double> &vec)
{
	vec.clear();
	ifstream file (filename.c_str(), ios::binary);

	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = MNIST_reverse_int(magic_number);

		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = MNIST_reverse_int(number_of_images);

		for(int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*) &temp, sizeof(temp));
			vec.push_back((double) temp);
		}
	}
}

/*
int main()
{
    string filename = "../data/MNIST_t10k-images.idx3-ubyte";
    int number_of_images = 10000;
    int image_size = 28 * 28;
    //read MNIST iamge into OpenCV Mat vector
    vector<cv::Mat> vec;
    read_Mnist(filename, vec);
    cout<<vec.size()<<endl;
    filename = "../data/MNIST_t10k-labels.idx1-ubyte";
    //read MNIST label into double vector
    vector<double> labels(number_of_images);
    read_Mnist_Label(filename, labels);
    cout<<labels.size()<<endl;
    for (int i = 0; i < vec.size(); i++)
    {
    	cout<<labels[i]<<endl;
		imshow("1st", vec[i]);
		waitKey();
    }
    return 0;
}
 */


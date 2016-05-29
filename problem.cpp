#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
#include <fstream>
using namespace std;

#include "xnn.h"
#include "profile.h"

namespace nn {
namespace impl {


//
// !! 
// some hardcoded paths to resources, fit to your needs !
//
String path_digits = "c:/p/data/img/digits.png"; // this is digits.png from opencv samples
String path_mnist = "c:/p/data/mnist";           // took it from tiny-cnn
String path_att = "c:/p/data/faces";             // std att-faces
String path_tv10 = "c:/p/data/tv10";             // my own set, probably has license issues.
//
//

Mat deskew(const Mat &img)
{
    PROFILEX("deskew");
    int SZ = img.rows;
    Moments m = moments(img);
    if (abs(m.mu02) < 1e-2)
        return img; 
    double skew = m.mu11 / m.mu02;
    Mat_<float> M(2,3); M << 1, skew, -0.5*SZ*skew, 0, 1, 0;
    Mat res;
    warpAffine(img, res, M, Size(SZ,SZ), WARP_INVERSE_MAP | INTER_LINEAR);
    return res;
}

struct Numbers : Problem
{
    const int ROWS = 1;
    const int COLS = 3;
    const int CLASSES = 2;

    virtual void train(int n, Volume &data, Volume &labels) {batch(n,data,labels);}
    virtual void test(int n, Volume &data, Volume &labels) {batch(n,data,labels);}
    virtual void batch(int n, Volume &data, Volume &labels) 
    {
        for (int i=0; i<n; i++)
        {
            PROFILEX("numbers_batch");
            UMat m(ROWS, COLS, CV_32F);
            Mat_<float> mc = m.getMat(ACCESS_WRITE);
            for (int j=0; j<COLS; j++)
            {
                mc(0,j) = theRNG().uniform(0.0f, 1.0f);
            }
            data.push_back(m);

            UMat lab(ROWS, CLASSES, CV_32F, 0.0f);
            Mat_<float> lc = lab.getMat(ACCESS_WRITE);
            int truth = (mc(0) > 0.5f) && (mc(1) > 0.5f);
            lc(0,truth) = 1.0f;
            labels.push_back(lab);
        }
    }
    virtual Size inputSize() { return Size(ROWS,COLS); } 
    virtual Size outputSize() { return Size(ROWS,CLASSES); }
    virtual String desc() {return format("Numbers(%d,%d,%d)",ROWS,COLS,CLASSES);}
};

struct Spiral : Problem
{
    const int CLASSES = 3;
    const int N=1000;

    UMat X, Y;

    // spiral
    Spiral()
    {
        Mat x,y;
        int K=CLASSES;
        int ix=0;
        x.create(N*K, 2, CV_32F);
        y.create(N*K, K, CV_32F);
        y.setTo(0);
        for (int k=0; k<K; k++)
        for (int n=1; n<=N; n++)
        {
            float r = float(N)/(n);
            float t = k*4 + (k+1)*4*r + theRNG().uniform(0.0f, N*0.2f);
            x.at<float>(ix,0) = r*sin(t); 
            x.at<float>(ix,1) = r*cos(t);
            y.at<float>(ix,k) = 1.0f;
            ix++;
        }
        x.copyTo(X);
        y.copyTo(Y);
        cerr << X.size() << " " << Y.size() << endl;
    } 
    virtual void train(int n, Volume &data, Volume &labels) {batch(n,data,labels, 0);}
    virtual void test(int n, Volume &data, Volume &labels) {batch(n,data,labels, N/2);}
    virtual void batch(int n, Volume &data, Volume &labels, int offset) 
    {
        for (int i=0; i<n; i++)
        {
            int idx = theRNG().uniform(0,N/2+offset);
            data.push_back(X.row(idx));
            labels.push_back(Y.row(idx));
        }
    }
    virtual Size inputSize() { return Size(1,2); } 
    virtual Size outputSize() { return Size(1,3); }
    virtual String desc() {return "Spiral(2,3,1000)";}
};

struct Digits : Problem
{
    const int SZ = 20;
    UMat digi;

    Digits()
    {
        PROFILE;
        Mat img = imread(path_digits, 0);
        for (int r=0; r<img.rows; r+=SZ)
        {
            for (int c=0; c<img.cols; c+=SZ)
            {
                Mat d = img(Rect(c, r, SZ,SZ));
                d = deskew(d); // in-place.
            }
        }
        img.convertTo(digi, CV_32F, 1.0/255.0);
    }
    virtual void train(int n, Volume &data, Volume &labels) {batch(n,data,labels,0);}
    virtual void test(int n, Volume &data, Volume &labels) {batch(n,data,labels,50);}
    virtual void batch(int n, Volume &data, Volume &labels, int off) 
    {
        for (int i=0; i<n; i++)
        {
            PROFILEX("digi_img");
            int r = theRNG().uniform(0,50);
            int c = theRNG().uniform(0,50) + off;
    
            UMat m;
            m = digi(Rect(c*SZ,r*SZ,SZ,SZ));
            data.push_back(m.clone());
    
            UMat lab(1,10,CV_32F,0.0f);
            Mat l=lab.getMat(ACCESS_WRITE);
            l.at<float>(int(r/5)) = 1.0f;
            labels.push_back(lab);
        }
    }
    virtual Size inputSize() { return Size(SZ,SZ); } 
    virtual Size outputSize() { return Size(1,10); } 
    virtual String desc() {return format("Digits(%dx%d)",SZ,SZ);}
};


struct AttFaces : Problem
{
    const int pSiz = 30;
    const int nPers = 40;
    const int nImg = 10;
    virtual void train(int n, Volume &data, Volume &labels) {batch(n,data,labels,0);}
    virtual void test(int n, Volume &data, Volume &labels) {batch(n,data,labels,5);}
    virtual void batch(int n, Volume &data, Volume &labels, int off) 
    {
        for (int i=0; i<n; i++)
        {
            int p = theRNG().uniform(1, 1+nPers);
            int n = theRNG().uniform(1, 1+(nImg/2)) + off;
            String fn = format("%s/s%d/%d.pgm", path_att.c_str(), p, n);
            Mat m = imread(fn, 0);
            if (m.empty())
            {
                cout << "bad " << fn << endl;
                continue;
            }
            UMat um;
            resize(m,m,Size(pSiz,pSiz));
            m.convertTo(um, CV_32F, 1.0/255.0);
            data.push_back(um);

            UMat lab(1,nPers,CV_32F,0.0f);
            Mat l=lab.getMat(ACCESS_WRITE);
            l.at<float>(p-1) = 1.0f;
            labels.push_back(lab);
        }
    }
    virtual Size inputSize() { return Size(pSiz,pSiz); } 
    virtual Size outputSize() { return Size(1,nPers); } 
    virtual String desc() {return format("AttFaces(%d,%d,%d))",pSiz,nPers,nImg);}
};

struct Tv10Faces : Problem
{
    const int pSiz  = 30;
    const int nPers = 30;
    const int nImg  = 10;
    vector<String> fn;

    Tv10Faces()
    {
        glob(path_tv10 + "/*.png", fn);
    }
    virtual void train(int n, Volume &data, Volume &labels) {batch(n,data,labels,0);}
    virtual void test(int n, Volume &data, Volume &labels) {batch(n,data,labels,5);}
    virtual void batch(int n, Volume &data, Volume &labels, int off) 
    {
        for (int i=0; i<n; i++)
        {
            int p = theRNG().uniform(0,nPers);
            int n = theRNG().uniform(0,(nImg/2)) + off;
            String f = fn[p*nImg+n];
            Mat m = imread(f, 0);
            if (m.empty())
            {
                cout << "bad " << f << endl;
                continue;
            }   
            UMat um;
            resize(m,m,Size(pSiz,pSiz));
            m.convertTo(um, CV_32F, 1.0/255.0);
            data.push_back(um);

            UMat lab(1,nPers,CV_32F,0.0f);
            Mat l=lab.getMat(ACCESS_WRITE);
            l.at<float>(p) = 1.0f;
            labels.push_back(lab);
        }
    }
    virtual Size inputSize() { return Size(pSiz,pSiz); } 
    virtual Size outputSize() { return Size(1,nPers); } 
    virtual String desc() {return format("Tv10Faces(%d,%d,%d))",pSiz,nPers,nImg);}
};

struct MNist : Problem
{
    vector<Mat> vec;
    vector<char> lab;

    MNist() // todo: currently, we're only using the testset.
    {
        read_images(path_mnist + "/t10k-images.idx3-ubyte");
        read_labels(path_mnist + "/t10k-labels.idx1-ubyte");
    }

    int reverse_int(int i)
    {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }

    void read_images(string filename)
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
            magic_number = reverse_int(magic_number);

            file.read((char*) &number_of_images,sizeof(number_of_images));
            number_of_images = reverse_int(number_of_images);

            file.read((char*) &n_rows, sizeof(n_rows));
            n_rows = reverse_int(n_rows);

            file.read((char*) &n_cols, sizeof(n_cols));
            n_cols = reverse_int(n_cols);

            for(int i = 0; i < number_of_images; ++i)
            {
                Mat_<uchar> tp(n_rows, n_cols);

                for(int r = 0; r < n_rows; ++r)
                {
                    for(int c = 0; c < n_cols; ++c)
                    {
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp(r, c) = temp;
                    }
                }
                vec.push_back(deskew(tp));
            }
        } 
        else clog << "bad input " << filename << endl;
    }

    void read_labels(string filename)
    {
        ifstream file (filename.c_str(), ios::binary);

        if (file.is_open())
        {
            int magic_number = 0;
            int number_of_images = 0;

            file.read((char*) &magic_number, sizeof(magic_number));
            magic_number = reverse_int(magic_number);

            file.read((char*) &number_of_images,sizeof(number_of_images));
            number_of_images = reverse_int(number_of_images);

            for(int i = 0; i < number_of_images; ++i)
            {
                unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
                lab.push_back(temp);
            }
        }
    }

    virtual void train(int n, Volume &data, Volume &labels) { batch(n, data, labels, 0); }
    virtual void test(int n, Volume &data, Volume &labels)  { batch(n, data, labels, vec.size() / 2); }
    virtual void batch(int n, Volume &data, Volume &labels, int off) 
    {
        for (int i=0; i<n; i++)
        {
            int id = theRNG().uniform(0, vec.size()/2) + off;
            Mat m = vec[id];

            UMat um;
            m.convertTo(um, CV_32F, 1.0/255.0);
            data.push_back(um);

            UMat lbl(1, 10, CV_32F,0.0f);
            Mat  l = lbl.getMat(ACCESS_WRITE);

            l.at<float>(lab[id]) = 1.0f;
            labels.push_back(lbl);
        }
    }
    virtual Size inputSize()  { return Size(28, 28); } 
    virtual Size outputSize() { return Size(1, 10); } 
    virtual String desc() { return format("MNist(%d,%d,%d))", 28, 10, vec.size());}
};


} // namespace impl


Ptr<Problem> createProblem(String name)
{
    using namespace nn::impl;
    if (name=="tv10") return makePtr<impl::Tv10Faces>();
    if (name=="att") return makePtr<impl::AttFaces>();
    if (name=="digits") return makePtr<impl::Digits>();
    if (name=="mnist") return makePtr<impl::MNist>();
    if (name=="numbers") return makePtr<impl::Numbers>();
    if (name=="spiral") return makePtr<impl::Spiral>();
}

} // namespace nn


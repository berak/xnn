#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

#include "profile.h"
#include <omp.h>

typedef vector<Mat> Volume;
typedef Mat (*proc)(const Mat &);

struct Problem
{
    virtual void train(int n, Volume &data, Volume &labels) = 0;
    virtual void test(int n, Volume &data, Volume &labels) = 0;
    virtual Size inputSize() = 0; 
    virtual Size outputSize() = 0; 
};

struct Layer
{
    virtual float forward(const Volume &upstream, Volume &downstream, bool training) = 0;
    virtual float backward(Volume &upstream, const Volume &downstream, float learn) = 0;
};

Mat rand(int r, int c) { Mat m(r,c,CV_32F); randu(m,0,1); return m; }
Mat minmax(const Mat &m) {normalize(m,m,1,0,NORM_MINMAX); return m; }

struct Fully : Layer
{
    Volume cache;
    Mat weights;

    virtual float forward(const Volume &upstream, Volume &downstream, bool training) 
    {   PROFILEX("full_fw");
        if (training)
            cache = upstream;
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            Mat up = upstream[i].reshape(1,1);
            Mat dn = up * weights;
            downstream[i] = dn;
        }
        return 0;
    }
    virtual float backward(Volume &upstream, const Volume &downstream, float learn)
    {   PROFILEX("full_bw");
        Mat wt = weights.t();
        upstream.resize(downstream.size());
        Mat grad(weights.size(), weights.type(), 0.0f);
        #pragma omp parallel for
        for (size_t i=0; i<downstream.size(); i++)
        {
            Mat dn = downstream[i];
            Mat up = dn * wt;
            upstream[i] = up;

            Mat c = cache[i].reshape(1,1);
            Mat pred = c * weights;
            Mat res = pred - dn;
            Mat dx = c.t() * res;
            grad += dx;
        }
        grad /= downstream.size();
        weights -= grad * learn;
        return sum(abs(grad))[0];
    }
};

struct Activation : Layer
{
    proc fw,bw;
    Activation(proc fw, proc bw) : fw(fw), bw(bw) {}

    virtual float forward(const Volume &upstream, Volume &downstream, bool training) 
    {   PROFILEX("act_fw");
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            downstream[i] = fw(upstream[i]);
        }
        return 0;
    }
    virtual float backward(Volume &upstream, const Volume &downstream, float learn)
    {   PROFILEX("act_bw");
        upstream.resize(downstream.size());
        for (size_t i=0; i<downstream.size(); i++)
        {
            upstream[i] = bw(downstream[i]);
        }
        return 0;
    }
};

struct Digits : Problem
{
    Mat digi;
    Digits()
    {
        digi = imread("digits.png", 0);
        digi.convertTo(digi, CV_32F, 1.0/255.0);
    }
    Mat deskew(const Mat &img)
    {
        PROFILE;
        Moments m = moments(img);
        if (abs(m.mu02) < 1e-2)
            return img; 
        double skew = m.mu11 / m.mu02;
        Mat_<float> M(2,3); M << 1, skew, -0.5*20*skew, 0, 1, 0;
        Mat res;
        warpAffine(img, res, M, Size(20,20), WARP_INVERSE_MAP | INTER_LINEAR);
        return res;
    }
    virtual void train(int n, Volume &data, Volume &labels) {batch(n,data,labels,0);}
    virtual void test(int n, Volume &data, Volume &labels) {batch(n,data,labels,50);}
    virtual void batch(int n, Volume &data, Volume &labels, int off) 
    {
        for (int i=0; i<n; i++)
        {
            int r = theRNG().uniform(0,50);
            int c = theRNG().uniform(0,50) + off;
            Mat m = digi(Rect(c*20,r*20,20,20));
            data.push_back(deskew(m));

            Mat_<float> lab(1,10,0.0f);
            lab(r/5) = 1.0f;
            labels.push_back(lab);
        }
    }
    virtual Size inputSize() { return Size(20,20); } 
    virtual Size outputSize() { return Size(1,10); } 
};

struct Faces : Problem
{
    const int pSiz = 30;
    virtual void train(int n, Volume &data, Volume &labels) {batch(n,data,labels,0);}
    virtual void test(int n, Volume &data, Volume &labels) {batch(n,data,labels,5);}
    virtual void batch(int n, Volume &data, Volume &labels, int off) 
    {
        for (int i=0; i<n; i++)
        {
            int p = theRNG().uniform(1,41);
            int n = theRNG().uniform(1,6) + off;
            String fn = format("c:/p/data/faces/s%d/%d.pgm",p,n);
            Mat m = imread(fn, 0);
            if (m.empty())
            {
                cerr << "bad " << fn << endl;
                continue;
            }
            resize(m,m,Size(pSiz,pSiz));
            m.convertTo(m, CV_32F, 1.0/255.0);
            data.push_back(m);

            Mat_<float> lab(1,40,0.0f);
            lab(p-1) = 1.0f;
            labels.push_back(lab);
        }
    }
    virtual Size inputSize() { return Size(pSiz,pSiz); } 
    virtual Size outputSize() { return Size(1,40); } 
};

struct tv10 : Problem
{
    const int pSiz = 25;
    const int nPers = 30;
    vector<String> fn;

    tv10()
    {
        glob("c:/p/data/tv10/*.png", fn);
    }
    virtual void train(int n, Volume &data, Volume &labels) {batch(n,data,labels,0);}
    virtual void test(int n, Volume &data, Volume &labels) {batch(n,data,labels,5);}
    virtual void batch(int n, Volume &data, Volume &labels, int off) 
    {
        for (int i=0; i<n; i++)
        {
            int p = theRNG().uniform(0,nPers);
            int n = theRNG().uniform(0,5) + off;
            String f = fn[p*10+n];
            Mat m = imread(f, 0);
            if (m.empty())
            {
                cerr << "bad " << f << endl;
                continue;
            }
            resize(m,m,Size(pSiz,pSiz));
            m.convertTo(m, CV_32F, 1.0/255.0);
            data.push_back(m);

            Mat_<float> lab(1,nPers,0.0f);
            lab(p) = 1.0f;
            labels.push_back(lab);
        }
    }
    virtual Size inputSize() { return Size(pSiz,pSiz); } 
    virtual Size outputSize() { return Size(1,nPers); } 
};

Mat viz(const Volume &v, int patchSize)
{
    int n = (int)sqrt(double(v.size())) + 1;
    Mat draw(n*patchSize, n*patchSize, CV_32F, 0.0f);
    for (size_t i=0; i<v.size(); i++)
    {
        Mat m = v[i].reshape(1,patchSize);
        int r = patchSize * (i / n);
        int c = patchSize * (i % n);
        m.copyTo(draw(Rect(c,r,patchSize,patchSize)));
    }
    return draw;
}

Mat viz(const Mat &weights)
{
    int pn = (int)sqrt(double(weights.cols)) + 1;
    int ps = (int)sqrt(double(weights.rows));
    Mat draw(pn*ps+2,pn*ps+2,CV_32F,0.0f);
    for (int i=0; i<weights.cols; i++)
    {
        Mat f = weights.col(i).clone().reshape(1,ps);
        f = minmax(f);
        int r = ps * int(i / pn);
        int c = ps * int(i % pn);
        //cerr << draw.size() << " " << pn  << " " << ps  << " " << r << " " << c << endl;
        f.copyTo(draw(Rect(c,r,ps,ps)));
    }
    return draw;
}

int main()
{
    PROFILE;
    //theRNG().state = getTickCount();
    Digits problem;
    int l1 = problem.inputSize().area(); 
    int l2 = 14*14;
    int l3 = problem.outputSize().area(); 
    Fully full1, full2;
    full1.weights = rand(l1,l2);
    full2.weights = rand(l2,l3);
    Activation act(minmax,minmax);
    for (int g=1; g<1000; g++)
    {
        Volume data,res,res1,res2,res3,labels;
        problem.train(32,data,labels);
        float e0 = full1.forward(data,res1,true);
        float x0 = act.forward(res1,res2,true);
        float e1 = full2.forward(res2,res,true);
        float e2 = full2.backward(res3,labels,0.001f);
        float x2 = act.backward(res2,res3,0.001f);
        float e3 = full1.backward(res1,res2,0.001f);
        
        //float e2 = full1.forward(data,res,true);
        //float e3 = full1.backward(res2,labels,0.01f);
        
        if (g%50==0)
        {
            float acc = 0.0f;
            for (size_t i=0; i<res.size(); i++)
            {
                Mat r1; labels[i].convertTo(r1,CV_32S);
                Mat r2; res[i].convertTo(r2,CV_32S);
                int ok = countNonZero(r1==r2);
                acc += float(ok) / r1.total();
            }
            cerr << g << " " << e2 << " " << e3 << " " << acc / res.size() << endl;
            namedWindow("input", 0);
            namedWindow("full1", 0);
            namedWindow("back", 0);
            imshow("full1", viz(full1.weights));
            imshow("back", viz(res1, problem.inputSize().width));
            imshow("middle", viz(res2, 14));
            imshow("input", viz(data, problem.inputSize().width));
            imshow("full2", viz(full2.weights));
            if (waitKey(1)==27) return 0;
        }
    }

    Volume data,res,res1,res2,labels;
    problem.test(100,data,labels);
    
    float e0 = full1.forward(data,res1,false);
    float x0 = act.forward(res1,res2,false);
    float e1 = full2.forward(res2,res,false);
    
    //full1.forward(data,res,true);
    float acc = 0.0f;
    for (size_t i=0; i<res.size(); i++)
    {
        Mat r1; labels[i].convertTo(r1,CV_32S);
        Mat r2; res[i].convertTo(r2,CV_32S);
        int ok = countNonZero(r1==r2);
        acc += float(ok) / r1.total();
    }
    cerr << "final: " << acc / res.size() << endl;
    waitKey();
    return 0;
}
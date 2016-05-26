#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

#include "xnn.h"
#include "profile.h"

namespace nn {
namespace impl {


typedef UMat (*proc)(const UMat &);

UMat identity(const UMat &m)
{
    return m;
}

UMat minmax(const UMat &m)
{
    PROFILE;
    normalize(m,m,1,0,NORM_MINMAX);
    return m; 
}
UMat mean(const UMat &m)
{
    PROFILE;
    Scalar _m, _s;
    meanStdDev(m, _m, _s);
    UMat res;
    subtract(m, _m, res);
    return res; 
}

UMat relu(const UMat &m)
{
    PROFILE;
    UMat u;
    max(m, 0, u);
    return u;
}
UMat relu_bp(const UMat &m)
{
    PROFILE;
    UMat mask;
    compare(m, 0, mask, CMP_GT);
    UMat u = m;
    u.setTo(1, mask);
    return u;
}

UMat sigmoid(const UMat &m)
{
    PROFILE;
    UMat P;
    multiply(m, -1, P);
    exp(P,P);
    add(P, 1, P);
    divide(1,P, P);
    return P;
}
UMat sigmoid_bp(const UMat &m)
{
    PROFILE;
    UMat res;
    subtract(1.0, m, res);
    res = res.mul(m);
    return res;
}

UMat tanh_fw(const UMat &m)
{
    PROFILE;
    UMat _m,ep,en,sp,sn,res;
    multiply(m, -1, _m);
    exp(m, ep);
    exp(_m, en);
    add(ep, en, sp);
    subtract(ep, en, sn);
    divide(sn, sp, res);
    return res;
}
UMat tanh_bw(const UMat &m)
{
    PROFILE;
    UMat _m, res;
    subtract(1, m, _m);
    sqrt(_m, res);
    return res;
}

UMat softmax(const UMat &m)
{
    UMat prob;
    double maxVal=0;
    minMaxLoc(m, 0, &maxVal);
    subtract(m, maxVal, prob);
    exp(prob, prob);
    Scalar total = sum(prob);
    divide(prob, total[0], prob);
    return prob;  
}

UMat rand(int r, int c)
{
    PROFILE;
    UMat m(r,c,CV_32F);
    randn(m,0.0,0.25);
    //randn(m,0,sqrt(2.0/(r*c)));
    return m;
}

UMat dropout(const UMat &m)
{
    PROFILE;
    float prob = 0.5f;
    UMat m1 = rand(m.rows, m.cols);
    UMat mask;
    compare(m1, prob, mask, CMP_GT);
    UMat res;
    multiply(m, mask, res, prob/255, CV_32F);
    return res;
}

struct Fully : Layer
{
    Volume cache_up, cache_dn;
    UMat weights;
    float learn;

    virtual float forward(const Volume &upstream, Volume &downstream, bool training) 
    {   
        PROFILEX("fully_forward");
        if (training)
        {
            cache_up = upstream;
            cache_dn.resize(upstream.size());
        }
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            UMat up = upstream[i].reshape(1,1);
            // dn = up * weights
            UMat dn;
            gemm(up, weights, 1, noArray(), 0, dn);
            downstream[i] = dn;
            if (training) cache_dn[i] = dn;
        }
        return 0;
    }

    virtual float backward(Volume &upstream, const Volume &downstream)
    {   
        PROFILEX("fully_backward");
        UMat wt = weights.t();
        upstream.resize(downstream.size());
        UMat grad(weights.size(), weights.type(), 0.0f);
        //#pragma omp parallel for
        for (size_t i=0; i<downstream.size(); i++)
        {
            UMat dn = downstream[i];
            // up = dn * wt;
            UMat up;
            gemm(dn, wt, 1, noArray(), 0, up);
            upstream[i] = up;
            // residual = predicted - truth
            UMat c = cache_up[i].reshape(1,1);
            UMat pred = cache_dn[i];
            UMat res;
            subtract(pred, dn, res);
            // dx = c.t() * res;
            UMat dx;
            gemm(c, res, 1, noArray(), 0, dx, GEMM_1_T);
            // grad += dx;
            add(grad, dx, grad);
        }
        // grad /= downstream.size();
        // weights -= grad * learn;
        scaleAdd(grad, -learn/downstream.size(), weights, weights);
        Mat grad_cpu; grad.copyTo(grad_cpu);
        return sum(abs(grad_cpu))[0];
    }

    virtual bool write(FileStorage &fs) 
    {
        Mat m = weights.getMat(ACCESS_READ);
        fs << "size" << weights.size();
        fs << "weights" << m;
        fs << "learn" << learn;
        return true;
    }
    virtual bool read(const FileNode &fn) 
    {
        Size siz;
        fn["size"] >> siz;
        Mat m = weights.getMat(ACCESS_WRITE);
        fn["weights"] >> m;
        if (m.empty() && siz.area())
        {
            weights = rand(siz.height, siz.width);
        }
        fn["learn"] >> learn;
        return true;
    }    
    virtual String type() {return "fully";}
    virtual String desc() {return format("fully(%dx%d)",weights.cols, weights.rows);}
    virtual void show(String winName)
    {
        namedWindow(winName);
        imshow(winName,viz(weights));
    }
};

struct RBM : Layer
{
    UMat hidden;
    UMat weights, weights_t;
    Volume cache_up, cache_dn, cache_hidden;
    float learn;

    UMat dream_wake(const UMat &m)
    {
        PROFILEX("rbm_dream_wake")
        UMat dream;
        gemm(m, weights, 1, noArray(), 0, dream);
        hidden = tanh_fw(dream);
        //hidden = sigmoid(dream);

        UMat wake;
        gemm(hidden, weights_t, 1, noArray(), 0, wake);
        return tanh_fw(wake);
        //return sigmoid(wake);
    }
    virtual float forward(const Volume &upstream, Volume &downstream, bool training) 
    {
        PROFILEX("rbm_forward")
        if (training)
        {
            cache_hidden.resize(upstream.size());
        }
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            UMat up = upstream[i].reshape(1,1);
            UMat dn = dream_wake(up);
            if (training)
                cache_hidden[i] = hidden;
            downstream[i] = dn;
        }
        if (training)
        {
            cache_up = upstream;
            cache_dn = downstream;
        }
    }
    virtual float backward(Volume &upstream, const Volume &downstream)
    {
        PROFILEX("rbm_backward")
        UMat grad(weights.size(), weights.type(), 0.0f);
        upstream.resize(downstream.size());
        for (size_t i=0; i<downstream.size(); i++)
        {
            UMat dn = downstream[i];
            UMat up = dream_wake(dn);
            upstream[i] = up;

            UMat dx1;
            gemm(dn, cache_hidden[i], 1, noArray(), 0, dx1, GEMM_1_T);
            UMat dx2;
            gemm(cache_dn[i].reshape(1,1), hidden, 1, noArray(), 0, dx2, GEMM_1_T);

            UMat dx;
            subtract(dx2, dx1, dx);
            add(grad, dx, grad);
        }
        scaleAdd(grad, -learn/downstream.size(), weights, weights);
        transpose(weights, weights_t);
    }
    virtual String type() { return "rbm"; }
    virtual String desc() { return format("rbm(%d,%d)",weights.cols,weights.rows); }
    virtual bool write(FileStorage &fs) 
    {
        Mat m = weights.getMat(ACCESS_READ);
        fs << "size" << weights.size();
        fs << "weights" << m;
        fs << "learn" << learn;
        return true;
    }
    virtual bool read(const FileNode &fn) 
    {
        Size siz;
        fn["size"] >> siz;
        Mat m = weights.getMat(ACCESS_WRITE);
        fn["weights"] >> m;
        if (m.empty() && siz.area())
        {
            weights = rand(siz.height, siz.width);
        }
        weights_t = weights.t();
        fn["learn"] >> learn;
        return true;
    }    
    virtual void show(String winName)
    {
        namedWindow(winName+"_weights");
        imshow(winName+"_weights", viz(weights));
        namedWindow(winName+"_hidden");
        imshow(winName+"_hidden", viz(cache_hidden, int(sqrt(double(hidden.cols)))));
    }
};


struct Activation : Layer
{
    proc fw,bw;
    String _n;
    Activation(proc fw, proc bw, String n="chicken") : fw(fw), bw(bw), _n(n) {}

    float pipe(const Volume &from, Volume &to, proc act) 
    {
        to.resize(from.size());
        for (size_t i=0; i<from.size(); i++)  to[i] = act(from[i]);
        return 0;
    }

    virtual float forward(const Volume &upstream, Volume &downstream, bool)
    {
        return pipe(upstream, downstream, fw);
    }
    virtual float backward(Volume &upstream, const Volume &downstream)
    {
        return pipe(downstream, upstream, bw);
    }
    virtual String type() {return _n;}
    virtual String desc() {return _n;}
};


struct Dropout : Layer
{
    virtual float forward(const Volume &upstream, Volume &downstream, bool training)
    {
        if (! training)
        {
            downstream = upstream;
            return 0;
        }
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++) 
            downstream[i] = dropout(upstream[i]);
        return 0;
    }
    virtual float backward(Volume &upstream, const Volume &downstream)
    {
        upstream = downstream;
        return 0;
    }
    virtual String type() { return "dropout"; }
    virtual String desc() { return "dropout"; }
};


struct XNN : Network
{
    vector<Ptr<Layer>> layers;

    virtual String desc() 
    { 
        String d="";
        for (size_t i=0; i<layers.size(); i++)
        {
            d += layers[i]->desc() + "\r\n";
        }
        return d; 
    }
    virtual float forward(const Volume &up, Volume &dn, bool training)
    {
        Volume a(up), b;
        for (size_t i=0; i<layers.size(); i++)
        {
            layers[i]->forward(a,b, training);
            cv::swap(a,b);
        }
        dn = a;
        return 0;
    }
    virtual float backward(Volume &up, const Volume &dn)
    {
        float e=0;
        Volume a, b(dn);
        for (int i=int(layers.size())-1; i>=0; i--)
        {
            e = layers[i]->backward(a,b);
            cv::swap(a,b);
        }
        up = b;
        return e; // 1st layer
    }
    virtual bool save(String  fn)
    {
        FileStorage fs(fn,1);
        if (!fs.isOpened())
        {
            clog << "could not save to " << fn << endl;
            return false;            
        }
        fs << "layers" << "[";
        for (size_t i=0; i<layers.size(); i++)
        {
            fs << "{:";
            fs << "type" << layers[i]->type();
            layers[i]->write(fs);
            fs << "}";
        }
        fs << "]";
        fs.release();
        return true;
    }
    virtual bool load(String  fn)
    {
        FileStorage fs(fn,0);
        if (!fs.isOpened())
        {
            clog << "could not load " << fn << endl;
            return false;
        }
        FileNode no = fs["layers"];
        for (FileNodeIterator it=no.begin(); it!=no.end(); ++it)
        {
            const FileNode &n = *it;
            String type;
            n["type"] >> type;
            Ptr<Layer> layer;
            if (type=="fully")    layer = makePtr<Fully>();
            if (type=="rbm")      layer = makePtr<RBM>();
            if (type=="sigmoid")  layer = makePtr<Activation>(sigmoid,sigmoid_bp,"sigmoid");
            if (type=="relu")     layer = makePtr<Activation>(relu,relu_bp,"relu");
            if (type=="tanh")     layer = makePtr<Activation>(tanh_fw,tanh_bw,"tanh");
            if (type=="mean")     layer = makePtr<Activation>(mean,mean,"mean");
            if (type=="minmax")   layer = makePtr<Activation>(minmax,minmax,"minmax");
            if (type=="dropout")  layer = makePtr<Dropout>();
            if (layer.empty())
            {
                clog << "unknown layer: " << type << endl;
                return false;
            }
            layer->read(n);
            layers.push_back(layer);
        }    
        fs.release();
        return true;
    }
    virtual void show()
    {
        for (size_t i=0; i<layers.size(); i++)
        {
            String n = format("%s_%d", layers[i]->type().c_str(), i);
            layers[i]->show(n);
        }
    }
};

} // namespace impl


Ptr<Network> createNetwork(String name)
{
    Ptr<Network> nn = makePtr<impl::XNN>();
    nn->load(name);
    return nn;
}


// public helper:
Mat viz(const Volume &v, int patchSize)
{
    PROFILEX("viz_vol")
    int n = (int)sqrt(double(v.size()*2));
    Mat draw(n*patchSize, n*patchSize, CV_32F, 0.0f);
    for (size_t i=0; i<v.size(); i++)
    {
        Mat m = v[i].getMat(ACCESS_READ).reshape(1,patchSize);
        normalize(m,m,1,0,NORM_MINMAX);
        int r = patchSize * (i / n);
        int c = patchSize * (i % n);
        m.copyTo(draw(Rect(c,r,patchSize,patchSize)));
    }
    return draw;
}
Mat viz(const UMat &weights)
{
    PROFILEX("viz_weights")
    int pn = (int)sqrt(double(weights.cols)) + 1;
    int ps = (int)sqrt(double(weights.rows));
    Mat draw(pn*ps+2,pn*ps+2,CV_32F,0.0f);
    for (int i=0; i<weights.cols; i++)
    {
        Mat f = weights.getMat(ACCESS_READ).col(i).clone().reshape(1,ps);
        normalize(f,f,1,0,NORM_MINMAX);
        int r = ps * int(i / pn);
        int c = ps * int(i % pn);
        f.copyTo(draw(Rect(c,r,ps,ps)));
    }
    return draw;
}

} // namespace nn


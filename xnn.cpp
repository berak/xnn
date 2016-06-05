#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
#include <deque>
using namespace std;

#include "xnn.h"
#include "profile.h"

namespace nn {
namespace impl {


typedef UMat (*proc)(const UMat &);

#include "activation.cpp"
#include "optimizer.cpp"


struct LinearLoss
{
    void operator()(const UMat &predicted,const UMat &truth, UMat &res) const
    {
        PROFILEX("loss_linear");
        subtract(predicted,truth,res);
    }
};

struct SoftmaxLoss
{
    void operator()(const UMat &predicted,const UMat &truth, UMat &res) const
    {
        PROFILEX("loss_softmax");
        UMat prob, mask;

        exp(predicted, prob);
        Scalar total = sum(prob);
        divide(prob, total[0], prob);

        truth.convertTo(mask,CV_8U);
        subtract(prob, -1, res, mask);
    }
};


struct BaseWeights :  Layer
{
    UMat weights, weights_t, bias;
    float learn, weight_init;
    String name;
    BaseWeights(String n="fully"): name(n) {}

    virtual bool write(FileStorage &fs) 
    {
        Mat m = weights.getMat(ACCESS_READ);
        Mat b = bias.getMat(ACCESS_READ);
        fs << "size" << weights.size();
        fs << "weights" << m;
        fs << "bias" << b;
        fs << "learn" << learn;
        fs << "weight_init" << weight_init;
        return true;
    }
    virtual bool read(const FileNode &fn) 
    {
        Size siz;
        fn["size"] >> siz;
        Mat m,b;
        fn["weights"] >> m;
        fn["bias"] >> b;
        weight_init = 0.5f;
        fn["weight_init"] >> weight_init;
        if (m.empty() && siz.area())
        {
            weights = rand(siz.height, siz.width, weight_init);
            bias = UMat(siz.width, 1, CV_32F, 0.0f);
        }
        else
        {
            m.copyTo(weights);
            b.copyTo(bias);
        }
        weights_t = weights.t();
        fn["learn"] >> learn;
        return true;
    }    
    virtual String type() { return name; }
    virtual String desc() { return format("%s(%d,%d,%1.3f,%1.3f)",name.c_str(), weights.cols, weights.rows, learn, weight_init); }
};

template <typename Optimizer, typename Loss>
struct Fully : BaseWeights 
{
    Optimizer optim_w, optim_b;
    Loss loss;
    Volume cache_up, cache_dn;
    Fully(String n="fully"): BaseWeights(n) {}

    virtual float forward(const Volume &upstream, Volume &downstream, bool training) 
    {   
        PROFILEX("fully_forward");
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            // dn = up * weights + bias
            UMat up = upstream[i].reshape(1,1);
            gemm(up, weights, 1, bias, 1, downstream[i], GEMM_3_T);
        }
        if (training)
        {
            cache_up = upstream;
            cache_dn = downstream;
        }
        return 0;
    }

    virtual float backward(Volume &upstream, const Volume &downstream, bool training)
    {   
        PROFILEX("fully_backward");
        upstream.resize(downstream.size());
        UMat grad(weights.size(), weights.type(), 0.0f);
        UMat gradB(bias.size(), bias.type(), 0.0f);
        //#pragma omp parallel for
        for (size_t i=0; i<downstream.size(); i++)
        {
            UMat dn = downstream[i];
            // up = dn * wt;
            UMat up;
            gemm(dn, weights_t, 1, noArray(), 0, up);
            upstream[i] = up;
            if (!training)
                continue;
            // residual = predicted - truth
            UMat pred = cache_dn[i];
            UMat res;
            loss(pred, dn, res);
            // dx = c.t() * res;
            // grad += dx;
            UMat c = cache_up[i];
            c = c.reshape(1, c.total());
            gemm(c, res, 1, grad, 1, grad);
            // bias, too.
            UMat db;
            reduce(res, db, 0, REDUCE_SUM);
            /*//cerr << db.size() << d.size() << endl;
            add(db, d.reshape(1,d.total()), db);           
            UMat db;
            reduce(grad, db, 1, REDUCE_SUM);*/
            add(gradB, gradB, db.reshape(1,db.total()));
        }
        if (training)
        {
            // grad /= downstream.size();
            // weights -= grad * learn;
            optim_w(grad, weights, learn/downstream.size());
            optim_b(gradB, bias, learn/downstream.size());
            weights_t = weights.t();
        }
        return 0.0f;
    }

    virtual void show(String winName)
    {
        namedWindow(winName);
        imshow(winName,viz(weights));
    }
};

/*
template <typename Optimizer>
struct Rnn : BaseWeights 
{
    Optimizer optim_w, optim_u;
    deque<UMat> past;
    Volume cache_up, cache_dn;
    proc act_fw, act_bw;
    UMat U, U_t;
    Rnn(String n="Rnn"): BaseWeights(n) {}

    UMat signal(const UMat &W, const UMat &U, const UMat &x, const UMat &last, proc act)
    {
        // act(last*U + x*W);
        UMat r;
        gemm(x, W, 1, noArray(), 0, r);
        if (! last.empty())
        {
            gemm(last, U, 1, r, 1, r);
        }
        return act(r);
    }
    
    virtual float forward(const Volume &upstream, Volume &downstream, bool training) 
    {   
        PROFILEX("rnn_forward");
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            // dn = up * weights + bias
            UMat up = upstream[i].reshape(1,1);
            past.push_back(up);
            UMat dn; // empty for the most left(oldest)
            for (size_t h=0; h<past.size(); h++)
            {
                dn = signal(weights, U, past[h], dn, act_fw);
            }
            downstream[i] = dn;
            if (past.size()>=3)
                past.pop_front();
        }
        if (training)
        {
            cache_up = upstream;
            cache_dn = downstream;
        }
        return 0;
    }

    virtual float backward(Volume &upstream, const Volume &downstream, bool training)
    {   
        PROFILEX("rbm_backward");
        upstream.resize(downstream.size());
        UMat grad(weights.size(), weights.type(), 0.0f);
        UMat gradU(U.size(), U.type(), 0.0f);
        //#pragma omp parallel for
        for (size_t i=0; i<downstream.size(); i++)
        {
            UMat dn = downstream[i];
            // up = dn * wt;
            UMat up;
            gemm(dn, weights_t, 1, noArray(), 0, up);
            upstream[i] = up;
            if (! training)
                continue;
            // residual = predicted - truth
            UMat pred = cache_dn[i];
            UMat res;
            subtract(pred, dn, res);
            // dx = c.t() * res;
            // grad += dx;
            UMat c = cache_up[i];
            c = c.reshape(1, c.total());
            gemm(c, res, 1, grad, 1, grad);
        }
        if (! training)
            return 0.0f;
        // grad /= downstream.size();
        // weights -= grad * learn;
        //scaleAdd(grad, -learn/downstream.size(), weights, weights);
        optim_w(grad, weights, learn/downstream.size());
        weights_t = weights.t();
        return 0.0f;
    }

    virtual void show(String winName)
    {
        namedWindow(winName);
        imshow(winName,viz(weights));
        namedWindow(winName+"U");
        imshow(winName+"U",viz(U));
    }
    virtual bool write(FileStorage &fs) 
    {
        BaseWeights::write(fs);
        Mat u = U.getMat(ACCESS_READ);
        fs << "U" << u;
        return true;
    }
    virtual bool read(const FileNode &fn) 
    {
        BaseWeights::read(fn);
        Mat u;
        fn["U"] >> u;
        if (u.empty())
        {
            U = rand(weights.cols, weights.rows, 0.01);
            U_t = U.t();
        }
        else
        {
            u.copyTo(U);
        }
        return true;
    }    
};

//
// https://cs231n.github.io/neural-networks-case-study/#linear
//  (minus the loss calculation)
//
template <typename Optimizer>
struct Softmax : BaseWeights
{
    Optimizer optim_w, optim_b;
    Volume cache_up, cache_dn;
    float reg;
    Softmax(String n="softmax"): BaseWeights(n) {}

    virtual float forward(const Volume &upstream, Volume &downstream, bool training) 
    {   
        PROFILEX("softmax_forward");
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            // dn = up * weights
            // cerr << up.size() << weights.size() << bias.size() << endl;          
            UMat up = upstream[i].reshape(1,1);
            gemm(up, weights, 1, bias, 1, downstream[i], GEMM_3_T);        
        }
        if (training)
        {
            cache_up = upstream;
            cache_dn = downstream;
        }
        return 0;
    }

    virtual float backward(Volume &upstream, const Volume &downstream, bool training)
    {   
        PROFILEX("softmax_backward");
        upstream.resize(downstream.size());
        UMat grad(weights.size(), weights.type(), 0.0f);
        UMat db(bias.size(), bias.type(), 0.0f);
        //#pragma omp parallel for
        for (size_t i=0; i<downstream.size(); i++)
        {
            UMat dn = downstream[i];
            // up = dn * wt + bias;
            UMat up;
            gemm(dn, weights_t, 1, noArray(), 0, up);

            upstream[i] = up;
            if (! training)
                continue;

            UMat pred = cache_dn[i];
            UMat prob, rsum;
            exp(pred, prob);
            Scalar total = sum(prob);
            divide(prob, total[0], prob);

            UMat mask,res;
            dn.convertTo(mask,CV_8U);
            subtract(prob, -1, res, mask);
            // dx = c.t() * res;
            // grad += dx;
            UMat c = cache_up[i];
            c = c.reshape(1, c.total());
            gemm(c, res, 1.0/downstream.size(), grad, 1, grad);

            UMat d;
            reduce(prob, d, 0, REDUCE_SUM);
            //cerr << db.size() << d.size() << endl;
            add(db, d.reshape(1,d.total()), db);           
            //add(db, d.t(), db);           
        }
        // grad /= downstream.size();
        // weights -= grad * learn;
        //scaleAdd(grad, -learn/downstream.size(), weights, weights);
        
        //float reg = 0.05;
        //scaleAdd(weights, reg, grad, grad);
        if (training)
        {
            optim_w(grad, weights, learn/downstream.size());
            optim_b(db, bias, learn/downstream.size());
            weights_t = weights.t();
        }
        return 0.0f;
    }

    virtual void show(String winName)
    {
        namedWindow(winName);
        imshow(winName,viz(weights));
    }
};


template <typename Optimizer>
struct RBM : BaseWeights
{
    Optimizer optim;
    UMat hidden;
    Volume cache_up, cache_dn, cache_hidden;
  
    RBM(String n="rbm"): BaseWeights(n) {}
    UMat dream_wake(const UMat &m)
    {
        PROFILEX("rbm_dream_wake")
        UMat dream;
        gemm(m, weights, 1, noArray(), 0, dream);
        hidden = tanh2_fw(dream);
        //hidden = sigmoid(dream);

        UMat wake;
        gemm(hidden, weights_t, 1, noArray(), 0, wake);
        return tanh2_fw(wake);
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
    virtual float backward(Volume &upstream, const Volume &downstream, bool training)
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
            //gemm(up, hidden, 1, noArray(), 0, dx2, GEMM_1_T);
            gemm(cache_up[i].reshape(1,1), hidden, 1, noArray(), 0, dx2, GEMM_1_T);

            UMat dx;
            subtract(dx1, dx2, dx);
            add(grad, dx, grad);
        }
        optim(grad, weights, learn/downstream.size());
        transpose(weights, weights_t);
    }
    virtual String type() { return "rbm"; }
    virtual void show(String winName)
    {
        namedWindow(winName+"_weights");
        imshow(winName+"_weights", viz(weights));
        namedWindow(winName+"_hidden");
        imshow(winName+"_hidden", viz(cache_hidden, int(sqrt(double(hidden.cols)))));
    }
};

*/

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
    virtual float backward(Volume &upstream, const Volume &downstream, bool training)
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
    virtual float backward(Volume &upstream, const Volume &downstream, bool training)
    {
        upstream = downstream;
        return 0;
    }
    virtual String type() { return "dropout"; }
    virtual String desc() { return "dropout"; }
};


struct BatchNorm : Layer
{
    virtual float batch(const Volume &from, Volume &to)
    {
        PROFILEX("batchnorm")
        to.resize(from.size());
        Scalar m,s, M, S;
        for (size_t i=0; i<from.size(); i++)
        {
            meanStdDev(from[0], m, s);
            M += m;
            S += s;
        }
        M[0] /= from.size();
        S[0] /= from.size();
        for (size_t i=0; i<from.size(); i++)
        {
            subtract(from[0], M, to[i]);
            divide(to[i], S[0]+0.0000001, to[i]);
        }
        /*UMat m(from[0].size(), from[0].type(), 0.0f);
        for (size_t i=0; i<from.size(); i++)
            add(from[0], m, m);
        divide(m, from.size(), m);

        UMat v(from[0].size(), from[0].type(), 0.00000001f);
        for (size_t i=0; i<from.size(); i++) 
        {
            UMat s;
            subtract(from[0], m, s);
            multiply(s,s,s);
            add(v,s,v);
        }
        divide(v, from.size(), v);
        for (size_t i=0; i<from.size(); i++) 
        {
            subtract(from[0], m, to[i]);
            divide(to[i], v, to[i]);
        }*/
        return 0;
    }
    virtual float forward(const Volume &upstream, Volume &downstream, bool training)
    {
        if (! training)
        {
            downstream = upstream;
            return 0;
        }
        return batch(upstream, downstream);
    }
    virtual float backward(Volume &upstream, const Volume &downstream, bool training)
    {
        if (! training)
        {
            upstream = downstream;
            return 0;
        }
        return batch(downstream, upstream);
    }
    virtual String type() { return "batchnorm"; }
    virtual String desc() { return "batchnorm"; }
};


struct XNN : Network
{
    vector<Ptr<Layer>> layers;
    String name;

    virtual String desc() 
    { 
        String d=format("%s %d gens\r\n", name.c_str(), ngens);
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
    virtual float backward(Volume &up, const Volume &dn, bool training)
    {
        ngens ++;
        float e=0;
        Volume a, b(dn);
        for (int i=int(layers.size())-1; i>=0; i--)
        {
            e = layers[i]->backward(a,b,training);
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
        fs << "ngens" << ngens;
        fs.release();
        return true;
    }
    virtual bool load(String  fn)
    {

        FileStorage fs(fn,0);
        if (!fs.isOpened())
        {
            CV_Error(0, String("could not load ") + fn);
        }
        name = fn;
        FileNode no = fs["layers"];
        int i=0;
        for (FileNodeIterator it=no.begin(); it!=no.end(); ++it)
        {
            const FileNode &n = *it;
            String type;
            n["type"] >> type;
            Ptr<Layer> layer;
            if (type=="sigmoid")  layer = makePtr<Activation>(sigmoid,sigmoid_bp,"sigmoid");
            if (type=="relu")     layer = makePtr<Activation>(relu,relu_bp,"relu");
            if (type=="tanh")     layer = makePtr<Activation>(tanh_fw,tanh_bw,"tanh");
            if (type=="tanh2")    layer = makePtr<Activation>(tanh2_fw,tanh2_bw,"tanh2");
            if (type=="mean")     layer = makePtr<Activation>(mean,mean,"mean");
            if (type=="minmax")   layer = makePtr<Activation>(minmax,minmax,"minmax");
            if (type=="dropout")  layer = makePtr<Dropout>();
            if (type=="batchnorm")layer = makePtr<BatchNorm>();

            if (type=="fully")    layer = makePtr<Fully<SGD, LinearLoss>>("fully");
            if (type=="fully_mom")layer = makePtr<Fully<momentum, LinearLoss>>("fully_mom");
            if (type=="fully_ada")layer = makePtr<Fully<adagrad, LinearLoss>>("fully_ada");
            if (type=="fully_rms")layer = makePtr<Fully<RMSprop, LinearLoss>>("fully_rms");
            if (type=="softmax")  layer = makePtr<Fully<SGD,SoftmaxLoss>>("softmax");
            if (type=="softmax_mom") layer = makePtr<Fully<momentum,SoftmaxLoss>>("softmax_mom");
            if (type=="softmax_rms") layer = makePtr<Fully<RMSprop,SoftmaxLoss>>("softmax_rms");
            //if (type=="rbm")      layer = makePtr<RBM<SGD>>();
            //if (type=="rnn")      layer = makePtr<Rnn<SGD>>();
            if (layer.empty())
            {
                CV_Error(0, format("unknown layer(%d): ", i) + type);
            }
            layer->read(n);
            layers.push_back(layer);
            i++;
        }
        fs["ngens"] >> ngens;
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
    int ngens;
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
        Mat m2, m = v[i].getMat(ACCESS_READ).reshape(1,patchSize);
        normalize(m, m2, 1, 0, NORM_MINMAX);
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
        normalize(f, f, 1, 0, NORM_MINMAX);
        int r = ps * int(i / pn);
        int c = ps * int(i % pn);
        f.copyTo(draw(Rect(c, r, ps, ps)));
    }
    return draw;
}

} // namespace nn


//
// all of it stolen from tiny-cnn..
//

struct SGD
{
    SGD(String n="") {}
    UMat operator()(const UMat &grad, UMat &weights, float learn)
    {
        PROFILEX("SGD");
        // W -= alpha * grad
        //cerr <<  "sgd:  " << grad.size() << " " << weights.size() << endl;
        scaleAdd(grad, -learn, weights, weights);
        return weights;
    }
    void write(FileStorage &fs) {}
    void read(const FileNode &fn) {}
};


struct momentum
{
    UMat G;
    String n;
    double mu;

    momentum(String n="") : n(n), mu(0.9) {}

    UMat operator()(const UMat &grad, UMat &weights, float learn)
    {
        PROFILEX("momentum");
        //    float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
        //    W[i]      += V;
        //    dWprev[i] =  V;

        if (G.empty())
            G=UMat(weights.size(), weights.type(), 0.0f);
        UMat V;
        multiply(G, mu, V);
        scaleAdd(grad, -learn, V, G);
        add(weights, G, weights);
        return weights;
    }
    void write(FileStorage &fs)
    {
        fs << n + "G" << G.getMat(ACCESS_READ);
    }
    void read(const FileNode &fn)
    {
        Mat g;
        fn[n + "G"] >> g;
        g.copyTo(G);
    }
};


// TODO: broken ?
struct adagrad
{
    adagrad(String n="", float e=1e-8f) : n(n), eps(e) {}

    UMat operator()(const UMat &grad, UMat &weights, float learn)
    {
        PROFILEX("adagrad");
        //    g[i] += dW[i] * dW[i];
        //    W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);

        if (G.empty())
            G=UMat(weights.size(), weights.type(), 0.0f);
        UMat a,b,c;
        multiply(grad, grad, a);
        add(G, a, G);
        sqrt(G, b);
        add(b, eps, b);
        divide(grad, b, c);
        scaleAdd(c, -learn, weights, weights);
        return weights;
    }
    void write(FileStorage &fs) {}
    void read(const FileNode &fn) {}

    String n;
    UMat G;
    float_t eps;
};


struct RMSprop
{
    RMSprop(String n, float m=0.99f, float e=1e-8f) : n(n), mu(m), eps(e) {}

    UMat operator()(const UMat &grad, UMat &weights, float learn)
    {
        PROFILEX("RMSProp");
        //    g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
        //    W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);

        if (G.empty())
            G=UMat(weights.size(), weights.type(), eps);
        UMat a,b,c;
        multiply(G, mu, G);
        multiply(grad, grad, a);
        scaleAdd(a, (1.0f-mu), G, G);
        sqrt(G, b);
        divide(grad, b, c);
        scaleAdd(c, -learn, weights, weights);
        return weights;
    }
    void write(FileStorage &fs)
    {
        fs << n + "G" << G.getMat(ACCESS_READ);
    }
    void read(const FileNode &fn)
    {
        Mat g;
        fn[n + "G"] >> g;
        g.copyTo(G);
    }

    String n;
    float_t mu; // decay term
    float_t eps; // constant value to avoid zero-division
    UMat G;
};

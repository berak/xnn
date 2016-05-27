// 
// all of it stolen from tiny-cnn..
//
struct SGD
{
    UMat operator()(const UMat &grad, UMat &weights, float learn)
    {
        PROFILEX("SGD");
        // W -= alpha * grad
        scaleAdd(grad, -learn, weights, weights);
        return weights;
    }
};

struct adagrad 
{
    adagrad(float e=1e-8f) : eps(e) {}

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

private:
    float_t eps;
    UMat G;
};

struct RMSprop
{
    RMSprop(float m=0.99f, float e=1e-8f) : mu(m), eps(e) {}

    UMat operator()(const UMat &grad, UMat &weights, float learn)
    {   
        PROFILES("RMSProp");
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

    float_t mu; // decay term
private:
    float_t eps; // constant value to avoid zero-division
    UMat G;
};




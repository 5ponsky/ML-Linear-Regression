#ifndef LAYER_H
#define LAYER_H

class Layer
{
	Vec activation;

public:
	Layer(size_t inputs, size_t outputs) :
	  activation(outputs)
	{
	}

	virtual ~Layer()
	{
	}

	virtual void activate(const Vec& weights, const Vec& x) = 0;
};

#endif //
#include "hmm.h"

const size_t DIGIT_NUM = 5;

/*************************************************
    Function(s)
*************************************************/
std::vector<Sequence> load_sequence(const std::string &filename)
{
    std::vector<Sequence> sequences;
    std::ifstream fin(filename.c_str());
    if (!fin.is_open())
    {
        throw "Couldn't load sequence: '" + filename + "'";
    }

    std::string line;
    while (std::getline(fin, line))
    {
        size_t length = line.size();
        Sequence sequence(length);
        for (size_t t = 0; t < length; ++t)
        {
            sequence[t] = line[t] - 'A';
        }

        sequences.push_back(sequence);
    }

    fin.close();
    return sequences;
}

/*************************************************
    Model - Static Method(s)
*************************************************/
Model Model::load(const std::string &filename)
{
    size_t state_num, observ_num;
    double *init_prob;
    double **trans_prob;
    double **observ_prob;

    std::ifstream fin(filename.c_str());
    if (!fin.is_open())
    {
        throw "Couldn't load model: '" + filename + "'";
    }

    std::string token;
    while (fin >> token, !fin.eof())
    {
        if (token == "initial:")
        {
            fin >> state_num;
            init_prob = new double[state_num];
            for (size_t i = 0; i < state_num; ++i)
            {
                fin >> init_prob[i];
            }
        }
        else if (token == "transition:")
        {
            fin >> state_num;
            trans_prob = new double*[state_num];
            for (size_t i = 0; i < state_num; ++i)
            {
                trans_prob[i] = new double[state_num];
                for (size_t j = 0; j < state_num; ++j)
                {
                    fin >> trans_prob[i][j];
                }
            }
        }
        else if (token == "observation:")
        {
            fin >> observ_num;
            observ_prob = new double*[observ_num];
            for (size_t o = 0; o < observ_num; ++o)
            {
                observ_prob[o] = new double[state_num];
                for (size_t i = 0; i < state_num; ++i)
                {
                    fin >> observ_prob[o][i];
                }
            }
        }
    }

    fin.close();

    // create model
    Model model(filename, state_num, observ_num, init_prob, trans_prob, observ_prob);

    // release variables
    for (size_t i = 0; i < state_num; ++i)
    {
        delete trans_prob[i];
    }

    for (size_t o = 0; o < observ_num; ++o)
    {
        delete observ_prob[o];
    }

    delete init_prob;
    delete trans_prob;
    delete observ_prob;

    return model;
}

void Model::dump(const std::string &filename, const Model &model)
{
    std::ofstream fout(filename.c_str());
    if (!fout.is_open())
    {
        throw "Couldn't dump model: '" + filename + "'";
    }

    fout << std::fixed << std::setprecision(DIGIT_NUM);
    fout << "initial: " << model._state_num << std::endl;
    for (size_t i = 0; i < model._state_num - 1; ++i)
    {
        fout << model.get_init_prob(i) << " ";
    }

    fout << model.get_init_prob(model._state_num - 1) << std::endl;

    fout << "\ntransition: " << model._state_num << std::endl;
    for (size_t i = 0; i < model._state_num; ++i)
    {
        for (size_t j = 0; j < model._state_num - 1; ++j)
        {
            fout << model.get_trans_prob(i, j) << " ";
        }

        fout << model.get_trans_prob(i, model._state_num - 1) << std::endl;
    }

    fout << "\nobservation: " << model._observ_num << std::endl;
    for (size_t o = 0; o < model._observ_num; ++o)
    {
        for (size_t i = 0; i < model._state_num - 1; ++i)
        {
            fout << model.get_observ_prob(o, i) << " ";
        }

        fout << model.get_observ_prob(o, model._state_num - 1) << std::endl;
    }

    fout.close();
}

/*************************************************
    Model - Constructor(s) & Destructor(s)
*************************************************/
Model::Model(const std::string &name, size_t state_num, size_t observ_num,
    double *init_prob, double **trans_prob, double **observ_prob)
    : _name(name), _state_num(state_num), _observ_num(observ_num)
{
    _malloc();

    set_init_prob(init_prob);
    set_trans_prob(trans_prob);
    set_observ_prob(observ_prob);
}

Model::Model(const Model &model) : _name(model._name),
    _state_num(model._state_num), _observ_num(model._observ_num)
{
    _malloc();

    set_init_prob(model._init_prob);
    set_trans_prob(model._trans_prob);
    set_observ_prob(model._observ_prob);
}

Model::~Model(void)
{
    _release();
}

/*************************************************
    Operator(s)
*************************************************/
Model &Model::operator=(const Model &model)
{
    // resize
    if (model._state_num != _state_num || model._observ_num != _observ_num)
    {
        _release();

        _state_num = model._state_num;
        _observ_num = model._observ_num;

        _malloc();
    }

    // copy probability
    set_init_prob(model._init_prob);
    set_trans_prob(model._trans_prob);
    set_observ_prob(model._observ_prob);

    return *this;
}

/*************************************************
    Model - Operate Method(s)
*************************************************/
double Model::evaluate(const Sequence &sequence) const
{
    size_t length = sequence.size();

    // calculate delta
    double *delta = new double[_state_num];
    double *new_delta = new double[_state_num];
    for (size_t i = 0; i < _state_num; ++i)
    {
        delta[i] = _init_prob[i] * _observ_prob[sequence[0]][i];
    }

    for (size_t t = 1; t < length; ++t)
    {
        for (size_t j = 0; j < _state_num; ++j)
        {
            double max_prob = delta[0] * _trans_prob[0][j];
            for (size_t i = 1; i < _state_num; ++i)
            {
                double prob = delta[i] * _trans_prob[i][j];
                if (prob > max_prob)
                {
                    max_prob = prob;
                }
            }

            new_delta[j] = max_prob * _observ_prob[sequence[t]][j];
        }

        memcpy(delta, new_delta, sizeof(double) * _state_num);
    }

    // calculate probability
    double prob = delta[0];
    for (size_t i = 1; i < _state_num; ++i)
    {
        if (delta[i] > prob)
        {
            prob = delta[i];
        }
    }

    delete delta;
    return prob;
}

void Model::learn(const std::vector<Sequence> &sequences, size_t iteration)
{
    size_t number = sequences.size();
    if (number == 0) { return; }

    size_t length = sequences[0].size();

    // allocate alpha, beta, gamma, and epsilon
    double **alpha = new double*[length];
    double **beta = new double*[length];
    double **gamma = new double*[length];
    double ***epsilon = new double**[length];

    double **sigma_gamma = new double*[length];
    double ***sigma_epsilon = new double**[length];

    for (size_t t = 0; t < length; ++t)
    {
        alpha[t] = new double[_state_num];
        beta[t]  = new double[_state_num];
        gamma[t]  = new double[_state_num];
        epsilon[t]  = new double*[_state_num];

        sigma_gamma[t]  = new double[_state_num];
        sigma_epsilon[t]  = new double*[_state_num];

        for (size_t s = 0; s < _state_num; ++s)
        {
            epsilon[t][s] = new double[_state_num];
            sigma_epsilon[t][s] = new double[_state_num];
        }
    }

    // allocate observ_gamma_sum
    double **observ_gamma_sum = new double*[_observ_num];
    for (size_t o = 0; o < _observ_num; ++o)
    {
        observ_gamma_sum[o] = new double[_state_num];
    }

    // perform learning
    for (size_t count = 0; count < iteration; ++count)
    {
        // initialize observ_gamma_sum, sigma_gamma and sigma_epsilon
        for (size_t o = 0; o < _observ_num; ++o)
        {
            memset(observ_gamma_sum[o], 0, sizeof(double) * _state_num);
        }

        for (size_t t = 0; t < length; ++t)
        {
            memset(sigma_gamma[t], 0, sizeof(double) * _state_num);
            for (size_t s = 0; s < _state_num; ++s)
            {
                memset(sigma_epsilon[t][s], 0, sizeof(double) * _state_num);
            }
        }

        for (std::vector<Sequence>::const_iterator iter = sequences.begin();
             iter != sequences.end(); ++iter)
        {
            const Sequence &sequence = *iter;

            // calculate alpha and beta
            _forward(alpha, sequence);
            _backward(beta, sequence);

            // calculate gamma
            for (size_t t = 0; t < length; ++t)
            {
                double prob_sum = 0;
                for (size_t i = 0; i < _state_num; ++i)
                {
                    gamma[t][i] = alpha[t][i] * beta[t][i];
                    prob_sum += gamma[t][i];
                }

                if (prob_sum == 0) { continue; }

                for (size_t i = 0; i < _state_num; ++i)
                {
                    gamma[t][i] /= prob_sum;
                    sigma_gamma[t][i] += gamma[t][i];
                    observ_gamma_sum[sequence[t]][i] += gamma[t][i];
                }
            }

            // calculate epsilon
            for (size_t t = 0; t < length - 1; ++t)
            {
                double prob_sum = 0;
                for (size_t i = 0; i < _state_num; ++i)
                {
                    for (size_t j = 0; j < _state_num; ++j)
                    {
                        epsilon[t][i][j] = alpha[t][i] \
                            * _trans_prob[i][j] \
                            * _observ_prob[sequence[t + 1]][j] \
                            * beta[t + 1][j];
                        prob_sum += epsilon[t][i][j];
                    }
                }

                if (prob_sum == 0) { continue; }

                for (size_t i = 0; i < _state_num; ++i)
                {
                    for (size_t j = 0; j < _state_num; ++j)
                    {
                        sigma_epsilon[t][i][j] += epsilon[t][i][j] / prob_sum;
                    }
                }
            }
        }

        for (size_t i = 0; i < _state_num; ++i)
        {
            // update initialize probability
            _init_prob[i] = sigma_gamma[0][i] / number;

            // update transition probability
            double gamma_sum = 0;
            for (size_t t = 0; t < length - 1; ++t)
            {
                gamma_sum += sigma_gamma[t][i];
            }

            if (gamma_sum > 0)
            {
                for (size_t j = 0; j < _state_num; ++j)
                {
                    double epsilon_sum = 0;
                    for (size_t t = 0; t < length - 1; ++t)
                    {
                        epsilon_sum += sigma_epsilon[t][i][j];
                    }

                    _trans_prob[i][j] = epsilon_sum / gamma_sum;
                }
            }
            else
            {
                memset(_trans_prob[i], 0, sizeof(double) * _state_num);
            }

            // update observation probability
            gamma_sum += sigma_gamma[length - 1][i];

            if (gamma_sum > 0)
            {
                for (size_t o = 0; o < _observ_num; ++o)
                {
                    _observ_prob[o][i] = observ_gamma_sum[o][i] / gamma_sum;
                }
            }
            else
            {
                for (size_t o = 0; o < _observ_num; ++o)
                {
                    _observ_prob[o][i] = 0;
                }
            }
        }
    }

    // release variables
    for (size_t o = 0; o < _observ_num; ++o)
    {
        delete observ_gamma_sum[o];
    }

    delete observ_gamma_sum;

    for (size_t t = 0; t < length; ++t)
    {
        for (size_t s = 0; s < _state_num; ++s)
        {
            delete sigma_epsilon[t][s];
            delete epsilon[t][s];
        }

        delete sigma_epsilon[t];
        delete sigma_gamma[t];

        delete epsilon[t];
        delete gamma[t];
        delete beta[t];
        delete alpha[t];
    }

    delete sigma_epsilon;
    delete sigma_gamma;

    delete epsilon;
    delete gamma;
    delete beta;
    delete alpha;
}

/*************************************************
    Model - Helper(s)
*************************************************/
void Model::_malloc(void)
{
    _init_prob = new double[_state_num];
    _trans_prob = new double*[_state_num];
    _observ_prob = new double*[_observ_num];

    for (size_t i = 0; i < _state_num; ++i)
    {
        _trans_prob[i] = new double[_state_num];
    }

    for (size_t o = 0; o < _observ_num; ++o)
    {
        _observ_prob[o] = new double[_state_num];
    }
}

void Model::_release(void)
{
    for (size_t i = 0; i < _state_num; ++i)
    {
        delete _trans_prob[i];
    }

    for (size_t o = 0; o < _observ_num; ++o)
    {
        delete _observ_prob[o];
    }

    delete _init_prob;
    delete _trans_prob;
    delete _observ_prob;
}

void Model::_forward(double **alpha, const Sequence &sequence) const
{
    size_t length = sequence.size();
    for (size_t i = 0; i < _state_num; ++i)
    {
        alpha[0][i] = _init_prob[i] * _observ_prob[sequence[0]][i];
    }

    for (size_t t = 0; t < length - 1; ++t)
    {
        for (size_t j = 0; j < _state_num; ++j)
        {
            double prob = 0;
            for (size_t i = 0; i < _state_num; ++i)
            {
                prob += alpha[t][i] * _trans_prob[i][j];
            }

            alpha[t + 1][j] = prob * _observ_prob[sequence[t + 1]][j];
        }
    }
}

void Model::_backward(double **beta, const Sequence &sequence) const
{
    size_t length = sequence.size();
    for (size_t i = 0; i < _state_num; ++i)
    {
        beta[length - 1][i] = 1;
    }

    for (size_t t = length - 1; t > 0; --t)
    {
        for (size_t i = 0; i < _state_num; ++i)
        {
            double prob = 0;
            for (size_t j = 0; j < _state_num; ++j)
            {
                prob += beta[t][j] * _trans_prob[i][j] * _observ_prob[sequence[t]][j];
            }

            beta[t - 1][i] = prob;
        }
    }
}


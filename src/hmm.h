#ifndef HMM_H_INCLUDED
#define HMM_H_INCLUDED

#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

/*************************************************
    Function(s)
*************************************************/
typedef std::vector<size_t> Sequence;
std::vector<Sequence> load_sequence(const std::string &filename);

/*************************************************
    Model - Class Declaration
*************************************************/
class Model
{
    public:
        // Static Method(s)
        static Model load(const std::string &filename);
        static void dump(const std::string &filename, const Model &model);

        // Constructor(s) & Destructor(s)
        Model(const std::string &name, size_t state_num, size_t observ_num,
            double *init_prob, double **trans_prob, double **observ_prob);
        Model(const Model &model);
        ~Model(void);

        // Operator(s)
        Model &operator=(const Model &model);

        // Setter(s) / Getter(s)
        std::string get_name(void) const;
        void        set_name(const std::string &name);
        size_t      get_state_num(void) const;
        size_t      get_observ_num(void) const;
        double      get_init_prob(size_t state) const;
        void        set_init_prob(double *prob);
        double      get_trans_prob(size_t state_from, size_t state_to) const;
        void        set_trans_prob(double **prob);
        double      get_observ_prob(size_t observ, size_t state) const;
        void        set_observ_prob(double **prob);

        // Operate Method(s)
        double      evaluate(const Sequence &sequence) const;
        void        learn(const std::vector<Sequence> &sequences, size_t iteration);

    private:
        // Helper(s)
        void        _malloc(void);
        void        _release(void);
        void        _forward(double **alpha, const Sequence &sequence) const;
        void        _backward(double **beta, const Sequence &sequence) const;

        // Data Member(s)
        std::string _name;
        size_t      _state_num;
        size_t      _observ_num;
        double      *_init_prob;
        double      **_trans_prob;
        double      **_observ_prob;
};

/*************************************************
    Model - Setter(s) / Getter(s)
*************************************************/
inline std::string Model::get_name(void) const
{
    return _name;
}

inline void Model::set_name(const std::string &name)
{
    _name = name;
}

inline size_t Model::get_state_num(void) const
{
    return _state_num;
}

inline size_t Model::get_observ_num(void) const
{
    return _observ_num;
}

inline double Model::get_init_prob(size_t state) const
{
    if (state >= _state_num) { return 0; }

    return _init_prob[state];
}

inline void Model::set_init_prob(double *prob)
{
    memcpy(_init_prob, prob, sizeof(double) * _state_num);
}

inline double Model::get_trans_prob(size_t state_from, size_t state_to) const
{
    if (state_from >= _state_num || state_to >= _state_num) { return 0; }

    return _trans_prob[state_from][state_to];
}

inline void Model::set_trans_prob(double **prob)
{
    for (size_t s = 0; s < _state_num; ++s)
    {
        memcpy(_trans_prob[s], prob[s], sizeof(double) * _state_num);
    }
}

inline double Model::get_observ_prob(size_t observ, size_t state) const
{
    if (state >= _state_num || observ >= _observ_num) { return 0; }

    return _observ_prob[observ][state];
}

inline void Model::set_observ_prob(double **prob)
{
    for (size_t o = 0; o < _observ_num; ++o)
    {
        memcpy(_observ_prob[o], prob[o], sizeof(double) * _state_num);
    }
}

#endif


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "hmm.h"

void error(const std::string &msg)
{
    std::cerr << msg << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    // argv[1]: model list path
    // argv[2]: test data path
    // argv[3]: result path
    if (argc < 4)
    {
        error("Usage: ./test model_list test_data output_result [answer]");
    }

    // load models
    std::vector<Model> model_list;
    std::ifstream fin(argv[1]);
    if (!fin.is_open())
    {
        error("Couldn't load model list: '" + std::string(argv[1]) + "'");
    }

    try
    {
        std::string filename;
        while (fin >> filename, !fin.eof())
        {
            Model model = Model::load(filename);
            model_list.push_back(model);
        }
    }
    catch (const std::string &msg)
    {
        error(msg);
    }

    fin.close();

    // load test data
    std::vector<Sequence> sequences = load_sequence(argv[2]);

    // write result
    std::ofstream fout(argv[3]);
    if (!fout.is_open())
    {
        error("Couldn't open result file: '" + std::string(argv[3]) + "'");
    }

    size_t seq_num = sequences.size();
    size_t model_num = model_list.size();
    std::vector<std::string> pred_list(seq_num);
    for (size_t n = 0; n < seq_num; ++n)
    {
        size_t sel = 0;
        double max_likelihood = model_list[0].evaluate(sequences[n]);
        for (size_t m = 1; m < model_num; ++m)
        {
            double likelihood = model_list[m].evaluate(sequences[n]);
            if (likelihood > max_likelihood)
            {
                sel = m;
                max_likelihood = likelihood;
            }
        }

        fout << model_list[sel].get_name() << '\t';
        fout << max_likelihood << std::endl;

        pred_list[n] = model_list[sel].get_name();
    }

    fout.close();

    if (argc >= 5)
    {
        fin.open(argv[4]);
        if (!fin.is_open())
        {
            error("Couldn't open result file: '" + std::string(argv[4]) + "'");
        }

        size_t count = 0;
        std::string ans;
        for (size_t n = 0; n < seq_num; ++n)
        {
            fin >> ans;
            if (ans == pred_list[n]) { ++count; }
        }

        fin.close();

        std::cout << static_cast<double>(count) / static_cast<double>(seq_num) << std::endl;
    }

    return EXIT_SUCCESS;
}


#include "wht-empvar-utils.h"

#include "nlohmann/json.hpp"

#include <iostream>

using json = nlohmann::json;
using namespace wht_empvar;

int main() {
    std::vector<float> x(64);
    for (int i = 0; i < 64; ++i) {
        x[i] = float((i % 7) - 3) / float(i + 3);
    }

    std::vector<std::vector<float>> calib = {
        x,
        normalize(turbo_rotate_inverse(x)),
        std::vector<float>(64),
        std::vector<float>(64),
    };

    for (int i = 0; i < 64; ++i) {
        calib[2][i] = float((i % 5) - 2) * 0.125f;
        calib[3][i] = float((i % 11) - 5) / 13.0f;
    }

    std::vector<std::vector<float>> eval = {
        calib[0],
        calib[2],
    };

    const auto normalized = normalize(x);
    const auto rotated = turbo_rotate_forward(normalized);
    const auto stats = empirical_variances(calib);
    std::vector<std::vector<float>> chunked_calib = {
        std::vector<float>(256),
        std::vector<float>(256),
        std::vector<float>(256),
    };
    for (int i = 0; i < 256; ++i) {
        chunked_calib[0][i] = std::sin(0.07f * float(i));
        chunked_calib[1][i] = std::cos(0.05f * float(i)) + 0.1f * float(i >= 128);
        chunked_calib[2][i] = float((i % 9) - 4) / 5.0f;
    }
    const auto chunked = empirical_variances_chunked(chunked_calib, 128);
    const auto emp_books = build_empirical_codebooks(stats.variances, 2);
    const auto fixed_books = build_fixed_codebooks(64, 2);
    const auto emp_recon = reconstruct_dataset(eval, emp_books, false);
    const auto fixed_recon = reconstruct_dataset(eval, fixed_books, true);

    json out;
    out["normalized"] = normalized;
    out["rotated"] = rotated;
    out["variances"] = stats.variances;
    out["chunked_variances"] = chunked.variances;
    out["empirical_recon_first"] = emp_recon.front();
    out["fixed_recon_first"] = fixed_recon.front();
    std::cout << out.dump(2) << "\n";
    return 0;
}

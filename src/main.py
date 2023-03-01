from models import SurrogateModel
from irace.compatibility.config_space import convert_from_config_space
from irace.parameters import Parameters
import pandas as pd

def main():
    m = SurrogateModel(
                    "../target_algorithms/surrogate/cplex_regions200/config_space.cplex_regions200.par10.random.pcs",
                    "../target_algorithms/surrogate/cplex_regions200/inst_feat_dict.cplex_regions200.par10.random.json",
                    "../target_algorithms/surrogate/cplex_regions200/pyrfr_model.cplex_regions200.par10.random.bin",
    )

    pcs: Parameters = convert_from_config_space(m.cs)
    pcs.guess_switch()
    
    df = pd.DataFrame.from_dict(pcs.parameters, orient='index', columns=("switch", "domain", "condition"))
    print(df.to_string())


if __name__ == '__main__':
    main()
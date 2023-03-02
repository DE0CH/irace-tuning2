from models import SurrogateModel
from irace.compatibility.config_space import convert_from_config_space
from irace.parameters import Parameters
import pandas as pd
from irace.constructors import get_irace_executable_path
import subprocess
from irace.constructors import make_scenario_args

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
    print(pcs.as_string())
    
    subprocess.run([get_irace_executable_path(), *make_scenario_args(dict(
        target_runner = "stdout://",
        parameter_file = "/dev/null",
        parameter_text = 'a "" i (0, 10) | TRUE',
        log_file = "/dev/null",
        train_instances_text = 'a\nb',
        train_instances_dir = "/",
        max_experiments = "48",
    ))])

if __name__ == '__main__':
    main()
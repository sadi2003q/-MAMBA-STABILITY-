



from config_loader import load_config
from seed_helper import set_seed
from data import build_data
from plain_predictor import PlainPredictor
from evaluation import one_step_error, chained_error


class configuration:
    plain_predictor = "plain_predictor"
    hidden_dim = "hidden_size"
    teacher_forcing_training = "teacher_forcing_training"
    epochs = "epochs"
    batch_size = "batch_size"
    lr = "learning_rate"

    windows = "windows"
    train = "train"
    test = "test"
    validation = "validation"
    normaliser = "normaliser"



config = load_config()
set_seed(config["model_seed"])




data = build_data()
model = PlainPredictor(
    config[configuration.plain_predictor][configuration.hidden_dim]
)
# model.load_state_dict(torch.load("outputs/stage3_plain.pt"))
states , push = data[configuration.windows][configuration.test]
normaliser = data[configuration.normaliser]
error_one_step = one_step_error(model, states, push, normaliser)
error_chained  =  chained_error(model, states, push, 10, 50, normaliser)


print(f"one step error : {error_one_step}")
print(f"chained error : {error_chained}")







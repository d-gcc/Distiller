# LightTS

Code for the paper LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation (under revision)

How to run the model:
 * Execute [model_distillation.py](model_distillation.py) specifying the model parameters, for example:
 `python model_distillation.py --experiment "NonInvasiveFetalECGThorax2" --evaluation "student" --teacher_type "Inception" --teachers 10 --distiller "kd" --learned_kl_w "True" --leaving_weights "True" --val_epochs 50 --gumbel 0.5 --bit1 4 --bit2 4 --bit3 4`
 * The complete list of parameters is available at lines 505--560 in [model_distillation.py](model_distillation.py).
 * Data is managed in [data.py](./utils/data.py). It follows the UCR archive structure, so the data sets folders are expected in `./dataset/TimeSeriesClassification/'
 * Results will be inserted in a database, connections are managed in [util.py](./utils/util.py).

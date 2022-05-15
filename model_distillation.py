#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse, torch, copy, os, time, numpy as np, pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from pathlib import Path
from utils.data import get_loaders, get_raw_data
from utils.util import str2bool, get_free_device 
from utils.inception import InceptionModel
from utils.distiller import DistillKL, KDEnsemble, TeacherWeights
from utils.trainer import train_single, train_distilled, validation, evaluate, evaluate_ensemble
from utils.CAWPE import train_probabilities

from ax import *
from ax.runners.synthetic import SyntheticRunner
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from botorch.models.gp_regression import SingleTaskGP
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
from plotly.offline import plot
from sktime.datatypes._panel._convert import from_2d_array_to_nested

import pickle


# In[3]:


def Run_NN_Teacher(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader, val_loader, test_loader = get_loaders(config)
    best_accuracy = 0
    start_training = time.time()
    
    for epoch in range(1, config.epochs + 1):
        train_single(epoch, train_loader, model, optimizer, config)
    
        if (epoch) % 100 == 0:
            training_time = time.time() - start_training
            current_accuracy = evaluate(test_loader, model, config, epoch, training_time)
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                if not os.path.exists('./teachers/'):
                    os.makedirs('./teachers/')
                model_name = f'Inception_{config.experiment}_{config.init_seed}_teacher.pkl'
                savepath = "./teachers/" + model_name
                torch.save(model.state_dict(), savepath)


# In[2]:


def Run_SK_Teacher(config):
    from sktime.datatypes._panel._convert import from_2d_array_to_nested
    training, testing = get_raw_data(config)
    
    X_train = from_2d_array_to_nested(training.x.squeeze().cpu().detach().numpy())
    y_train = training.y.squeeze().cpu().detach().numpy()
    
    X_test = from_2d_array_to_nested(testing.x.squeeze().cpu().detach().numpy())
    y_test = testing.y.squeeze().cpu().detach().numpy()

    if config.teacher_type == 'CIF':
        from sktime.classification.interval_based import CanonicalIntervalForest
        classifier = CanonicalIntervalForest(random_state=config.init_seed)
    elif config.teacher_type == 'Forest':
        from sktime.classification.interval_based import TimeSeriesForestClassifier
        classifier = TimeSeriesForestClassifier(random_state=config.init_seed)
    elif config.teacher_type == 'Proximity':
        from sktime.classification.distance_based import ProximityTree
        classifier = ProximityTree(random_state=config.init_seed)
    elif config.teacher_type == 'TDE':
        from sktime.classification.dictionary_based import IndividualTDE
        classifier = IndividualTDE(random_state=config.init_seed)
    elif config.teacher_type == 'Rocket':
        from sktime.classification.kernel_based import RocketClassifier
        classifier = RocketClassifier(random_state=config.init_seed)
    elif config.teacher_type == 'Matrix':
        from sktime.classification.feature_based import MatrixProfileClassifier
        classifier = MatrixProfileClassifier(random_state=config.init_seed)
    
    classifier.fit(X_train, y_train)
    
    model_name = f'{config.teacher_type}_{config.experiment}_{config.init_seed}_teacher.pkl'
    savepath = "./teachers/" + model_name
    
    with open(savepath,'wb') as file:
        pickle.dump(classifier,file)


# In[4]:


def RunStudent(model, config, teachers):
    config.teachers = len(teachers)
    config.teachers_removed = None
    try:
        config.teachers_removed = list(set(config.teacher_setting) - set(teachers))
    except:
        pass
    config.teacher_setting = teachers

    model_s = model
    model_s.eval()
    model_s = model_s.to(config.device)
    params = list((model_s.parameters()))

    module_list = nn.ModuleList([])
    module_list.append(model_s)

    criterion_list = nn.ModuleList([])
    criterion_list.append(nn.CrossEntropyLoss())
    
    if config.distiller == 'kd':
        criterion_list.append(DistillKL(config.kd_temperature))
    elif config.distiller == 'kd_baseline':
        criterion_list.append(KDEnsemble(config.kd_temperature, config.device))
    elif config.distiller == 'ae-kd':
        criterion_list.append(DistillKL(config.kd_temperature))
    elif config.distiller == 'cawpe':
        criterion_list.append(DistillKL(config.kd_temperature))
    elif config.distiller == 'kd_rl':
        criterion_list.append(DistillKL(config.kd_temperature))

    if config.teacher_type == 'Inception':
        train_loader, val_loader, test_loader = get_loaders(config)
    else:
        config.batch_size = 10000
        train_loader, val_loader, test_loader = get_loaders(config)
        
    # Teachers
    if config.teacher_type == 'Inception':
        for teacher in teachers:
            savepath = Path('./teachers/Inception_' + config.experiment + '_' + str(teacher) + '_teacher.pkl')
            teacher_config = copy.deepcopy(config)
            teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = config.bits
            teacher_config.layer1 = teacher_config.layer2 = teacher_config.layer3 = 3
            model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                           bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=config.num_classes,config=teacher_config)

            model_t.load_state_dict(torch.load(savepath, map_location=config.device))
            model_t.eval()
            model_t = model_t.to(config.device)
            module_list.append(model_t)
    else:
        teacher_list = []
        teacher_val_list = []
        for teacher in teachers:
            savepath = Path('./teachers/'+ config.teacher_type + '_' + config.experiment + '_' + str(teacher) + '_teacher.pkl')
            with open(savepath,'rb') as file:
                pickle_saved = pickle.load(file)
            
            for idx, data in enumerate(train_loader):
                input, target = data
                X_test = from_2d_array_to_nested(input.squeeze().cpu().detach().numpy())
                logit_t_np = pickle_saved.predict_proba(X_test)
                logit_t = torch.as_tensor(logit_t_np, dtype = torch.float, device = config.device)
                teacher_list.append(logit_t)

            for idx, data in enumerate(val_loader):
                input, target = data
                X_test = from_2d_array_to_nested(input.squeeze().cpu().detach().numpy())
                logit_t_np = pickle_saved.predict_proba(X_test)
                logit_t = torch.as_tensor(logit_t_np, dtype = torch.float, device = config.device)
                teacher_val_list.append(logit_t)
            
    if config.random_init_w:
        teacher_weights = torch.rand(config.teachers, device = config.device)
    elif config.specific_teachers:
        teacher_weights = torch.tensor([float(item) for item in config.list_weights.split(',')])
    else:
        teacher_weights = torch.full((1,config.teachers), 1/config.teachers, dtype=torch.float32, 
                                     device = config.device).squeeze()
    
    
    weights_model = TeacherWeights(config, teacher_weights)
    module_list.append(weights_model)
    params.extend(list(weights_model.parameters()))
    optimizer = torch.optim.Adam(model_s.parameters(), lr=config.lr)
    optimizer_w = torch.optim.SGD(weights_model.parameters(), lr=config.lr_w) #Adam ignores the bi-level
        
    module_list.to(config.device)
    criterion_list.to(config.device)
    

    start_training = time.time()
    
    if config.distiller == 'cawpe':
        config.evaluation = 'cross_validation'
        teacher_probs = train_probabilities(config)
        config.evaluation = 'student'
    elif config.distiller == 'kd_rl':
        teacher_probs = torch.full((1,config.teachers), 1/config.teachers, dtype=torch.float32, 
                                     device = config.device).squeeze()
                        #torch.rand(config.teachers, device = config.device)
    
    max_accuracy = accuracy = 0
    
    for epoch in range(1, config.epochs + 1):
        if config.distiller == 'cawpe':
            train_distilled(epoch, train_loader, module_list, criterion_list, optimizer, config, teacher_probs)
        elif config.distiller == 'kd_rl':
            reward = train_distilled(epoch, train_loader, module_list, criterion_list, optimizer, config, teacher_probs, t_list = teacher_list)
            teacher_probs -= reward * config.lr
            teacher_probs = torch.softmax(teacher_probs, dim=-1)
        else:
            train_distilled(epoch, train_loader, module_list, criterion_list, optimizer, config, t_list = teacher_list)
        
        if config.learned_kl_w and (epoch) % config.val_epochs == 0:
            teacher_weights = validation(epoch, val_loader, module_list, criterion_list, optimizer_w, config,t_list = teacher_val_list)
        if (epoch) % 100 == 0:
            training_time = time.time() - start_training
            accuracy = evaluate(test_loader, model_s, config, epoch, training_time)
        elif config.pid == 0:
            training_time = time.time() - start_training
            teacher_weights = validation(epoch, val_loader, module_list, criterion_list, optimizer_w, config,t_list = teacher_val_list)
            accuracy = evaluate(test_loader, model_s, config, epoch, training_time)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            
    return max_accuracy, dict(zip(teachers, teacher_weights))


# In[5]:


def remove_elements(x):
    return [[el for el in x if el!=x[i]] for i in range(len(x))]

def recursive_accuracy(model,config,max_accuracy,current_teachers):
    subgroups = remove_elements(current_teachers)
    for subgroup in subgroups:
        pivot_accuracy, _ = RunStudent(model, config, subgroup)
        if pivot_accuracy > max_accuracy:
            max_accuracy = pivot_accuracy
            if len(subgroup) > 2:
                recursive_accuracy(model,config,max_accuracy,subgroup)
    return max_accuracy # The value is not updated, so the recursivity continues

def recursive_weight(model,config,teacher_dic):
    
#     if config.gumbel > 0:
#         weights_tensor = torch.tensor(list(teacher_dic.values()))
#         weights_tensor.requires_grad = False
#         choice = F.gumbel_softmax(weights_tensor.mul(-1), tau = config.gumbel, dim=0)
#         weights_choice = dict(zip(list(teacher_dic.keys()), choice.tolist()))
#         ordered_weights = sorted(weights_choice.items(), key=lambda x: x[1], reverse=True)
#     else:
    ordered_weights = sorted(teacher_dic.items(), key=lambda x: x[1], reverse=False)
    
    if len(list(teacher_dic.keys())) > 8 and config.explore_branches > 1:
        for i in range(0,config.explore_branches):
            copy_weights = copy.deepcopy(teacher_dic)
            del copy_weights[ordered_weights[i][0]]
            new_teachers = list(copy_weights.keys())
            _, new_weights = RunStudent(model, config, new_teachers)
            accuracy = recursive_weight(model,config,new_weights)
    else:
        #remove_key = min(teacher_dic.keys(), key=lambda k: teacher_dic[k])
        del teacher_dic[ordered_weights[0][0]]
        new_teachers = list(teacher_dic.keys())
        accuracy, new_weights = RunStudent(model, config, new_teachers)
        if len(new_teachers) > 2:
            accuracy = recursive_weight(model,config,new_weights)
    return accuracy

def StudentDistillation(model, config):
    max_accuracy = pivot_accuracy = 0

    if config.specific_teachers:
        teachers = config.list_teachers
    else:    
        teachers = [i for i in range(0,config.teachers)]
    max_accuracy, teacher_weights = RunStudent(model, config, teachers)
    
    if config.distiller == 'kd':
        if config.leaving_out:
            max_accuracy = recursive_accuracy(model, config, max_accuracy, teachers)

        elif config.leaving_weights:
            max_accuracy = recursive_weight(model, config, teacher_weights)
        
    return max_accuracy

def TeacherEvaluation(config):
    if config.teacher_type == 'Inception':
        _, _, test_loader = get_loaders(config)
    else:
        config.batch_size = 10000
        _, _, test_loader = get_loaders(config)
    evaluate_ensemble(test_loader, config)


# In[6]:


class StudentBO():
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.init_teachers = self.config.teachers

    def __call__(self,params):
        self.config.bit1 = params["bit_1"]
        self.config.bit2 = params["bit_2"]
        self.config.bit3 = params["bit_3"]
        self.config.layer1 = params["layers_1"]
        self.config.layer2 = params["layers_2"]
        self.config.layer3 = params["layers_3"]
        max_accuracy = pivot_accuracy = 0

        if self.config.specific_teachers:
            teachers = config.list_teachers
        else:    
            teachers = [i for i in range(0,self.init_teachers)]
        max_accuracy, teacher_weights = RunStudent(self.model, self.config, teachers)

        if self.config.leaving_out:
            max_accuracy = recursive_accuracy(self.model, self.config, max_accuracy, teachers)

        if self.config.leaving_weights:
            max_accuracy = recursive_weight(self.model, self.config, teacher_weights)

        return max_accuracy

def build_experiment(search_space,optimization_config):
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment

def initialize_experiment(experiment,initialization):
    sobol = Models.SOBOL(search_space=experiment.search_space, seed=1234)

    for _ in range(initialization):
        trial = experiment.new_trial(sobol.gen(1))
        trial.run()
        trial.mark_completed()

    return experiment.fetch_data()


# In[7]:


class MetricAccuracy(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": student_bo(params),
                "sem": 0,
            })
        return Data(df=pd.DataFrame.from_records(records))
    
class MetricCost(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            bit_cost = params["layers_1"] * params["bit_1"] + params["layers_2"] * params["bit_2"] + params["layers_3"] * params["bit_3"]
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": bit_cost,
                "sem": 0,
            })
        return Data(df=pd.DataFrame.from_records(records))


# In[8]:


def BayesianOptimization(config):
    config.layer1 = config.layer2 = config.layer3 = 3
    model_s = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                   bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                   num_pred_classes=config.num_classes,config=config)
    model_s = model_s.to(config.device)
    student_bo = StudentBO(model_s, config)
    
    bit_1=ChoiceParameter(name="bit_1", values=[12,13,14], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    bit_2=ChoiceParameter(name="bit_2", values=[12,13,14], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    bit_3=ChoiceParameter(name="bit_3", values=[12,13,14], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    #layers_1=ChoiceParameter(name="layers_1", values=[3,4], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    #layers_2=ChoiceParameter(name="layers_2", values=[3,4], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    #layers_3=ChoiceParameter(name="layers_3", values=[3,4], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    layers_1=FixedParameter(name="layers_1", value=3, parameter_type=ParameterType.INT)
    layers_2=FixedParameter(name="layers_2", value=3, parameter_type=ParameterType.INT)
    layers_3=FixedParameter(name="layers_3", value=3, parameter_type=ParameterType.INT)


    search_space = SearchSpace(parameters=[bit_1, bit_2, bit_3, layers_1, layers_2, layers_3])
    
    metric_accuracy = GenericNoisyFunctionMetric("accuracy", f=student_bo, noise_sd=0.0, lower_is_better=False)

    #metric_accuracy2 = MetricAccuracy(name="accuracy2",lower_is_better=False)
    metric_cost = MetricCost(name="cost",lower_is_better=True)
    
    if config.evaluation == 'student_bo':
        objectives = MultiObjective(objectives=[Objective(metric=metric_accuracy), Objective(metric=metric_cost)])
        objective_thresholds = [
            ObjectiveThreshold(metric=metric_accuracy, bound=0.7, relative=False),
            ObjectiveThreshold(metric=metric_cost, bound=45, relative=False),
        ]

        optimization_config = MultiObjectiveOptimizationConfig(
            objective=objectives,
            objective_thresholds=objective_thresholds,
        )

        bo_experiment = build_experiment(search_space,optimization_config)
        bo_data = initialize_experiment(bo_experiment,config.bo_init)

        bo_model = None
        for i in range(config.bo_steps):
            bo_model = Models.MOO_MODULAR(
                experiment=bo_experiment, data=bo_data,
                surrogate=ListSurrogate(
                botorch_submodel_class_per_outcome={"accuracy": SingleTaskGP, "cost": SingleTaskGP,},
                submodel_options_per_outcome={"accuracy": {}, "cost": {}},))

            generator_run = bo_model.gen(1)
            params = generator_run.arms[0].parameters

            trial = bo_experiment.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()
            bo_data = Data.from_multiple_data([bo_data, trial.fetch_data()])
            
            exp_df = exp_to_df(bo_experiment)

            outcomes = np.array(exp_to_df(bo_experiment)[['accuracy', 'cost']], dtype=np.double)

            frontier = compute_posterior_pareto_frontier(
                experiment=bo_experiment,
                data=bo_experiment.fetch_data(),
                primary_objective=metric_accuracy,
                secondary_objective=metric_cost,
                absolute_metrics=["accuracy", "cost"],
                num_points=config.bo_init + config.bo_steps,
            )

            plot(plot_pareto_frontier(frontier, CI_level=0.90).data, filename=config.experiment+'_'+str(config.pid)+'_.html')
    
    elif config.evaluation == 'student_bo_simple':
        bo_experiment = SimpleExperiment(search_space=search_space,evaluation_function=student_bo)
        bo_experiment.runner = SyntheticRunner()
        config.bo_status = 'Random'
        bo_data = initialize_experiment(bo_experiment,config.bo_init)
        config.bo_status = 'Optimized'
        bo_model = None
        for i in range(config.bo_steps):
            bo_model = Models.BOTORCH(experiment=bo_experiment, data=bo_data)

            generator_run = bo_model.gen(1)
            params = generator_run.arms[0].parameters

            trial = bo_experiment.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()
            bo_data = Data.from_multiple_data([bo_data, trial.fetch_data()])

            exp_df = exp_to_df(bo_experiment)


# In[9]:


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="A dummy argument for Jupyter", default="1")
    parser.add_argument('--experiment', type=str, default='SyntheticControl') 

    # Quantization
    parser.add_argument('--bits', type=int, default=32)
    parser.add_argument('--bit1', type=int, default=13)
    parser.add_argument('--bit2', type=int, default=12)
    parser.add_argument('--bit3', type=int, default=8)
    parser.add_argument('--std_dev', type=float, default=0)
    parser.add_argument('--power_two', type=str2bool, default=False)
    parser.add_argument('--additive', type=str2bool, default=False)
    parser.add_argument('--grad_scale', type=int, default=None)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_w', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=1500)
    parser.add_argument('--init_seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--evaluation', type=str, default='student', 
                        choices=['teacher', 'student', 'teacher_ensemble', 'student_bo', 'student_bo_simple'])
    parser.add_argument('--teacher_type', type=str, default='Inception',
                        choices=['Inception', 'CIF', 'Forest', 'Proximity', 'TDE', 'Rocket', 'Matrix'])
    parser.add_argument('--bo_init', type=int, default=10)
    parser.add_argument('--bo_steps', type=int, default=10)
    
    # Distillation
    parser.add_argument('--distiller', type=str, default='kd', choices=['kd', 'kd_baseline','ae-kd','cawpe','kd_rl'])
    parser.add_argument('--kd_temperature', type=float, default=5)
    parser.add_argument('--teachers', type=int, default=10)

    parser.add_argument('--w_ce', type=float, default=1, help='weight for cross entropy')
    parser.add_argument('--w_kl', type=float, default=-1, help='weight for KL')
    parser.add_argument('--w_other', type=float, default=0.1, help='weight for other losses')
    
    # Leaving-out, learned weights
    parser.add_argument('--leaving_out', type=str2bool, default=False)
    parser.add_argument('--learned_kl_w', type=str2bool, default=False)
    parser.add_argument('--random_init_w', type=str2bool, default=False)
    parser.add_argument('--leaving_weights', type=str2bool, default=False)
    parser.add_argument('--avoid_mult', type=str2bool, default=False)
    parser.add_argument('--explore_branches', type=int, default=1)
    parser.add_argument('--val_epochs', type=int, default=1)
    parser.add_argument('--gumbel', type=float, default=1.0)
    parser.add_argument('--cross_validation', type=int, default=5)
    
    parser.add_argument('--specific_teachers', type=str2bool, default=False)
    parser.add_argument('--list_teachers', type=str, default="2,4,5,7,9")
    parser.add_argument('--list_weights', type=str, default="0.1,0.4,0.4")
    
    # SAX - PAA
    parser.add_argument('--use_sax', type=int, default=0)
    parser.add_argument('--sax_symbols', type=int, default=8)
    parser.add_argument('--paa_segments', type=int, default=10)
    
    config = parser.parse_args()
    config.list_teachers = [int(item) for item in config.list_teachers.split(',')]
    
    if config.device == -1:
        config.device = torch.device(get_free_device())
    else:
        config.device = torch.device("cuda:" + str(config.device))
    
    if config.init_seed > -1:
        np.random.seed(config.init_seed)
        torch.manual_seed(config.init_seed)
        torch.cuda.manual_seed(config.init_seed)
        torch.backends.cudnn.deterministic = True
        
    if config.distiller != 'kd':
        config.leaving_out = False
        config.learned_kl_w = False
        config.leaving_weights = False
        config.avoid_mult = False

    df = pd.read_csv('TimeSeries.csv',header=None)
    num_classes = int(df[(df == config.experiment).any(axis=1)][1])
    if num_classes == 2:
        num_classes = 1
    config.num_classes = num_classes

    if config.device == torch.device('cpu'):
        config.data_folder = Path('./dataset/TimeSeriesClassification')
    elif os.uname()[1] == 'cs-gpu04':
        config.data_folder = Path('/data/dgcc/TimeSeriesClassification')
    else:
        config.data_folder = Path('/data/cs.aau.dk/dgcc/TimeSeriesClassification')
        
    if config.evaluation == 'teacher' and config.teacher_type == 'Inception':
        teacher_config = config
        teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = config.bits
        teacher_config.layer1 = teacher_config.layer2 = teacher_config.layer3 = 3
        model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                       bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                       num_pred_classes=config.num_classes,config=teacher_config)
        model_t = model_t.to(config.device)
        
        for teacher in range(0,config.teachers):
            config.init_seed = teacher
            np.random.seed(teacher)
            torch.manual_seed(teacher)
            torch.cuda.manual_seed(teacher)
            torch.backends.cudnn.deterministic = True
            Run_NN_Teacher(model_t, config)
    elif config.evaluation == 'teacher' and config.teacher_type != 'Inception':
        for teacher in range(0,config.teachers):
            config.init_seed = teacher
            np.random.seed(teacher)
            torch.manual_seed(teacher)
            torch.cuda.manual_seed(teacher)
            torch.backends.cudnn.deterministic = True
            Run_SK_Teacher(config)
        TeacherEvaluation(config)
    elif config.evaluation == 'teacher_ensemble':
        TeacherEvaluation(config)
    elif config.evaluation == 'student_bo' or config.evaluation == 'student_bo_simple':
        BayesianOptimization(config)
    elif config.evaluation == 'student':
        config.layer1 = config.layer2 = config.layer3 = 3
        model_s = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                       bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                       num_pred_classes=config.num_classes,config=config)

        model_s = model_s.to(config.device)
        StudentDistillation(model_s, config)


# In[ ]:





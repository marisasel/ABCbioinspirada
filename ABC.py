__author__ = "Omur Sahin"

import sys
import csv
import numpy as np
from deap.benchmarks import *
import progressbar

class ABC:

    def __init__(_self, conf):
        _self.conf = conf
        #colocar nossos vetores de entrada
        _self.entrada = []
        _self.foods = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        _self.f = np.ones((_self.conf.FOOD_NUMBER))
        _self.fitness = np.ones((_self.conf.FOOD_NUMBER)) * np.iinfo(int).max
        _self.trial = np.zeros((_self.conf.FOOD_NUMBER))
        _self.prob = [0 for x in range(_self.conf.FOOD_NUMBER)]
        _self.solution = np.zeros((_self.conf.DIMENSION))
        _self.globalParams = [0 for x in range(_self.conf.DIMENSION)]
        _self.globalTime = 0
        _self.evalCount = 0
        _self.cycle = 0
        _self.experimentID = 0
        _self.globalOpts = list()

        if (_self.conf.SHOW_PROGRESS):
            _self.progressbar = progressbar.ProgressBar(max_value=_self.conf.MAXIMUM_EVALUATION)
        if (not(conf.RANDOM_SEED)):
            random.seed(conf.SEED)
    #Le csv de entrada
    def readCsv(_self):
        with open('entrada1.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            #Para cada linha
            for row in csv_reader:
                line_count += 1
                #Trata a lista de sucessores e antecessores
                sucessores = row[1].split(",")
                antecessores = row[2].split(",")
                for i in range(len(sucessores)):
                    if(sucessores[i] == "-"):
                        sucessores[i] = 0
                    else:
                        sucessores[i] = int(sucessores[i])
                #Em caso de lista nula atribui 0 já que as tarefas começam com 1
                #Este 0 é usado para controle na criação da população inicial
                for i in range(len(antecessores)):
                    if(antecessores[i] == "-"):
                        antecessores[i] = 0
                    else:
                        antecessores[i] = int(antecessores[i])
                #Cria a estrutura com os dados
                _self.entrada.append([int(row[0]),sucessores,antecessores,row[3],row[4],row[5],row[6],row[7]])
            print("Processed " +str(line_count)+ " lines.")

    def calculate_function(_self, sol):
        try:
            if (_self.conf.SHOW_PROGRESS):
                _self.progressbar.update(_self.evalCount)
            return _self.conf.OBJECTIVE_FUNCTION(sol)

        except ValueError as err:
            print(
                "An exception occured: Upper and Lower Bounds might be wrong. (" + str(err) + " in calculate_function)")
            sys.exit()

    def calculate_fitness(_self, fun):
        _self.increase_eval()
        if fun >= 0:
            result = 1 / (fun + 1)
        else:
            result = 1 + abs(fun)
        return result

    def increase_eval(_self):
        _self.evalCount += 1

    def stopping_condition(_self):
        status = bool(_self.evalCount >= _self.conf.MAXIMUM_EVALUATION)
        if(_self.conf.SHOW_PROGRESS):
          if(status == True and not( _self.progressbar._finished )):
               _self.progressbar.finish()
        return status

    def memorize_best_source(_self):
        for i in range(_self.conf.FOOD_NUMBER):
            if (_self.f[i] < _self.globalOpt and _self.conf.MINIMIZE == True) or (_self.f[i] >= _self.globalOpt and _self.conf.MINIMIZE == False):
                _self.globalOpt = np.copy(_self.f[i])
                _self.globalParams = np.copy(_self.foods[i][:])

    def init(_self, index):
        if (not (_self.stopping_condition())):
            i =0
            #Primeira parte: cria a lista de tarefa obedecendo as predecessoes
            #Segunda parte: atribui um processador de forma aleatoria
            while i < int(_self.conf.DIMENSION/2):
                ordenacao = []
                ordena = 1
                #Para cada tarefa lida do csv
                # 1º ordena todas as tarefas sem predecessores
                # 2º ordena as tarefas cujos predecessoes ja foram ordenados
                # Repete a etapa 2 ate que todas as tarefas sejam ordenadas
                for j in range(len(_self.entrada)):
                    #Tarefas sem predecessores
                    if(_self.entrada[j][2][0] == 0):
                        if((len(_self.foods[index]) >0) and (_self.entrada[j][0] not in _self.foods[index])) :
                            ordenacao.append(_self.entrada[j][0])
                        elif(len(_self.foods[index]) == 0):
                            ordenacao.append(_self.entrada[j][0])
                    #Tarefas que os predecessores ja foram ordenados
                    else:
                        for k in range(len(_self.entrada[j][2])):
                            if(_self.entrada[j][2][k] not in _self.foods[index]):
                                ordena = 0
                        if ((ordena) and (_self.entrada[j][0] not in _self.foods[index])):
                            ordenacao.append(_self.entrada[j][0])

                #Ordena de forma aleatoria
                while(len(ordenacao)):
                    r = random.choice(ordenacao)
                    while r not in ordenacao:
                        r = random.choice(ordenacao)
                    _self.foods[index][i] = r
                    ordenacao.remove(r)
                    i+=1

           #Atribui um processador de forma aleatoria 
            r = random.choice([1,2,3,4])
            _self.foods[index][10] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][11] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][12] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][13] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][14] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][15] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][16] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][17] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][18] = r
            r = random.choice([1,2,3,4])
            _self.foods[index][19] = r
            print(_self.foods[index])

            _self.solution = np.copy(_self.foods[index][:])
            _self.f[index] = _self.calculate_function(_self.solution)[0]
            _self.fitness[index] = _self.calculate_fitness(_self.f[index])
            _self.trial[index] = 0

    def initial(_self):
        print("Generating initial population")
        for i in range(_self.conf.FOOD_NUMBER):
            print("Food source: "+ str(i))
            _self.init(i)
        _self.globalOpt = np.copy(_self.f[0])
        _self.globalParams = np.copy(_self.foods[0][:])

    def send_employed_bees(_self):
        i = 0
        while (i < _self.conf.FOOD_NUMBER) and (not (_self.stopping_condition())):
            #Mudar para selecionar o intervalo de tarefas (Maximum size of..)
            #1 e 5
            #1 e 3
            #e a posicao delas no vetor de solucao
            # e qual dos vetores vai mudar
            r = random.random()
            _self.param2change = (int)(r * _self.conf.DIMENSION)

            #Seleciona o vizinho que vai visitar aleatoriamente
            r = random.random()
            _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
            while _self.neighbour == i:
                r = random.random()
                _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
            _self.solution = np.copy(_self.foods[i][:])


            ##essa e a parte que vamos colocar as operacoes de mudanca
            #implementar randomico para selecao da operacao 20% cada uma
            r = random.random()
            _self.solution[_self.param2change] = _self.foods[i][_self.param2change] + (
                        _self.foods[i][_self.param2change] - _self.foods[_self.neighbour][_self.param2change]) * (
                                                             r - 0.5) * 2
            #Controlar a regra dos sucessores
            #Nao precisa controlar o limite de valores pq nao alteramos os valores so a ordenacao
            if _self.solution[_self.param2change] < _self.conf.LOWER_BOUND:
                _self.solution[_self.param2change] = _self.conf.LOWER_BOUND
            if _self.solution[_self.param2change] > _self.conf.UPPER_BOUND:
                _self.solution[_self.param2change] = _self.conf.UPPER_BOUND

            #Continua igual, temos que alterar a calculate_function
            _self.ObjValSol = _self.calculate_function(_self.solution)[0]
            _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)
            #Continua igual
            if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                _self.trial[i] = 0
                _self.foods[i][:] = np.copy(_self.solution)
                _self.f[i] = _self.ObjValSol
                _self.fitness[i] = _self.FitnessSol
            else:
                _self.trial[i] = _self.trial[i] + 1
            i += 1

    def calculate_probabilities(_self):
        maxfit = np.copy(max(_self.fitness))
        for i in range(_self.conf.FOOD_NUMBER):
            _self.prob[i] = (0.9 * (_self.fitness[i] / maxfit)) + 0.1

    def send_onlooker_bees(_self):
        i = 0
        t = 0
        while (t < _self.conf.FOOD_NUMBER) and (not (_self.stopping_condition())):
            r = random.random()
            if ((r < _self.prob[i] and _self.conf.MINIMIZE == True) or (r > _self.prob[i] and _self.conf.MINIMIZE == False)):
                t+=1
                r = random.random()
                _self.param2change = (int)(r * _self.conf.DIMENSION)
                r = random.random()
                _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
                while _self.neighbour == i:
                    r = random.random()
                    _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
                _self.solution = np.copy(_self.foods[i][:])

                r = random.random()
                _self.solution[_self.param2change] = _self.foods[i][_self.param2change] + (
                            _self.foods[i][_self.param2change] - _self.foods[_self.neighbour][_self.param2change]) * (
                                                                 r - 0.5) * 2
                if _self.solution[_self.param2change] < _self.conf.LOWER_BOUND:
                    _self.solution[_self.param2change] = _self.conf.LOWER_BOUND
                if _self.solution[_self.param2change] > _self.conf.UPPER_BOUND:
                    _self.solution[_self.param2change] = _self.conf.UPPER_BOUND

                _self.ObjValSol = _self.calculate_function(_self.solution)[0]
                _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)
                if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                    _self.trial[i] = 0
                    _self.foods[i][:] = np.copy(_self.solution)
                    _self.f[i] = _self.ObjValSol
                    _self.fitness[i] = _self.FitnessSol
                else:
                    _self.trial[i] = _self.trial[i] + 1
            i += 1
            i = i % _self.conf.FOOD_NUMBER

    def send_scout_bees(_self):
        if np.amax(_self.trial) >= _self.conf.LIMIT:
            _self.init(_self.trial.argmax(axis = 0))

    def increase_cycle(_self):
        _self.globalOpts.append(_self.globalOpt)
        _self.cycle += 1
    def setExperimentID(_self,run,t):
        _self.experimentID = t+"-"+str(run)

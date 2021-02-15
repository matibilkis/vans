import pickle
from utilities.circuit_basics import Basic
import numpy as np
import os
from datetime import datetime

class Evaluator(Basic):
    def __init__(self, args, info=None, loading=False, acceptange_percentage = 0.01, accuracy_to_end=-np.inf,
                nrun_load=0, path="../data-vans/"):
        """
        This class serves as evaluating the energy, admiting the new circuit or not. Also stores the results either if there's a relevant modification or not.
        Finally, it allows for the possibilty of loading previous results, an example for the TFIM is:


        if_finish_ok: bool, if accuracy to end is reached, then VANS stops
        accuraccy to end: ground_state_energy (or estimation of it) + chemical_accuracy (or some percentage for cm hamiltonians)

        (cheat-sheet for jupyter notebook):
            %load_ext autoreload
            %autoreload 2
            from utilities.circuit_basics import Evaluator
            evaluator = Evaluator(loading=True, args={"n_qubits":3, "J":4.5})
            unitary, energy, indices, resolver = evaluator.raw_history[47]
            {"channel":"depolarizing", "channel_params":[p], "q_batch_size":10**3}

        """
        super(Evaluator, self).__init__(n_qubits=args["n_qubits"])
        self.path = path

        if not loading:
            self.raw_history = {}
            self.evolution = {}
            self.lowest_energy = None
            self.if_finish_ok = False
            self.accuracy_to_end = accuracy_to_end


            problem_identifier = self.get_problem_identifier(args["problem_config"])
            noise_identifier = self.get_noise_identifier(args["noise_config"])
            self.identifier = "{}/N{}_{}_{}".format(args["problem_config"]["problem"],args["n_qubits"],problem_identifier, noise_identifier)

            self.directory = self.create_folder(info)
            self.acceptange_percentage = acceptange_percentage
            self.displaying = "\n Hola, I'm VANS, and current local time is {} \n".format(datetime.now())
        else:
            args_load={}
            for str,default in zip(["n_qubits", "problem_config", "noise_config"], [4, {"problem":"TFIM", "g":1.0, "J": 0.0}, {}]):
                if str not in list(args.keys()):
                    args_load[str] = default
                else:
                    args_load[str] = args[str]
            problem_identifier = self.get_problem_identifier(args_load["problem_config"])
            noise_identifier = self.get_noise_identifier(args_load["noise_config"])
            self.identifier = "{}/N{}_{}_{}".format(args["problem_config"]["problem"],args_load["n_qubits"],problem_identifier, noise_identifier)
            self.load(args_load,nrun=nrun_load)

    def get_problem_identifier(self, args):
        #### read possible hamiltonians to get id structure
        with open("utilities/cm_hamiltonians.txt") as f:
            hams = f.readlines()
        cm_hamiltonians = [x.strip().upper() for x in hams]

        with open("utilities/chemical_hamiltonians.txt") as f:
            hams = f.readlines()
        chemical_hamiltonians=([x.strip().upper() for x in hams])

        if args["problem"].upper() in cm_hamiltonians:
            id = "g{}J{}".format(args["g"],args["g"])
        elif args["problem".upper()] in chemical_hamiltonians:
            id = "geometry_{}_multip_{}_charge_{}_basis{}".format(args["problem"], args["geometry"], args["multiplicity"], args["charge"], args["basis"])
        else:
            raise NameError("Check that your args.problem_config is correct")
        return id

    def get_noise_identifier(self, noise_config):
        if noise_config == {}:
            id = "quiet"
        else:
            id=""
            for key in ["channel"]:#"shots",
                if key in list(noise_config.keys()):
                    id+="{}_".format(key,noise_config[key])
            if id=="":
                raise NameError("could not get noise label from noise_config. Fed dict is {}".format(noise_config))
        return id

    def create_folder(self, info):
        """
        self.path is data-vans
        """
        name_folder=self.path+self.identifier
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
            nrun=0
            final_folder = name_folder+"/run_"+str(nrun)
            with open(name_folder+"/runs.txt", "w+") as f:
                f.write(info)
                f.close()
            os.makedirs(final_folder)
        else:
            folder = os.walk(name_folder)
            nrun=0
            for k in list(folder)[0][1]:
                if k[0]!=".":
                    nrun+=1
            final_folder = name_folder+"/run_"+str(nrun)
            with open(name_folder+"/runs.txt", "r") as f:
                a = f.readlines()[0]
                f.close()
            with open(name_folder+"/runs.txt", "w") as f:
                f.write(str(nrun)+"\n")
                f.write(info)
                f.write("\n")
                f.close()
            os.makedirs(final_folder)
        return final_folder

    def load(self,args, nrun=0):
        name_folder = self.path+self.identifier+"/run_{}".format(nrun)
        self.load_dicts_and_displaying(name_folder)
        return

    def save_dicts_and_displaying(self):
        output = open(self.directory+"/raw_history.pkl", "wb")
        pickle.dump(self.raw_history, output)
        output.close()
        output = open(self.directory+"/evolution.pkl", "wb")
        pickle.dump(self.evolution, output)
        output.close()
        with open(self.directory+"/evolution.txt","w") as f:
            f.write(self.displaying)
            f.close()
        return

    def load_dicts_and_displaying(self,folder, load_txt=False):
        with open(folder+"/raw_history.pkl" ,"rb") as h:
            self.raw_history = pickle.load(h)
        with open(folder+"/evolution.pkl", "rb") as hh:
            self.evolution = pickle.load(hh)
        if load_txt is True:
            with open(folder+"/evolution.txt", "r") as f:
               a = f.readlines()
               f.close()
            self.displaying = a
        return

    def accept_energy(self, E, noise=False):
        """
        in the presence of noise, don't give gates for free!
        E: energy after some optimization (to be accepted or not).
        For the moment we leave the same criteria for the noisy scenario also.
        """
        if self.lowest_energy is None:
            return True
        else:
            return (E-self.lowest_energy)/np.abs(self.lowest_energy) < self.acceptange_percentage

    def get_best_iteration(self):
        """
        returns minimum in evolution.
        """
        return list(np.where(np.array(list(self.evolution.values()))[:,1] == np.min(np.array(list(self.raw_history.values()))[:,-1]))[0])

    def number_cnots_best(self):
        cn=0
        for k in self.evolution[self.get_best_iteration()[0]][2]: #get the indices
            if k < self.number_of_cnots:
                cn+=1
        return cn

    def add_step(self,indices, resolver, energy, relevant=True):
        """
        indices: list of integers describing circuit to save
        resolver: dictionary with the corresponding circuit's symbols
        energy: expected value of target hamiltonian on prepared circuit.
        relevant: if energy was minimized on that step
        """
        if self.lowest_energy is None:
            self.lowest_energy = energy
        elif energy < self.lowest_energy:
            self.lowest_energy = energy

        if self.lowest_energy < self.accuracy_to_end:
            self.if_finish_ok = True

        self.raw_history[len(list(self.raw_history.keys()))] = [self.give_unitary(indices, resolver), energy, indices, resolver, self.lowest_energy]
        if relevant == True:
            self.evolution[len(list(self.evolution.keys()))] = [self.give_unitary(indices, resolver), energy, indices,resolver, self.lowest_energy]
        self.save_dicts_and_displaying()

        return

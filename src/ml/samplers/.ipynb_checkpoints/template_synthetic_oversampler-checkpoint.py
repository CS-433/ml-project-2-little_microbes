import logging
import numpy as np 
import pandas as pd
from typing import Tuple
from collections import Counter
from ml.samplers.final_utils import *

from imblearn.over_sampling import RandomOverSampler as ros
from ml.samplers.sampler import Sampler

class TemplateOversampler(Sampler):
    """
    This class oversamples some of the samples in the data depending on the sampling strategy chosen. 
    
    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'template oversampling'
        self._notation = 'tempos'
        
        self._rebalancing_mode = self._settings['ml']['oversampler']['rebalancing_mode'] 

    def _oversample(self, sequences:list, labels:list, oversampler:list, sampling_strategy:dict) -> Tuple[list, list, list]:
        """Oversamples some input sequences based on oversampler, according to the sampling_strategy.
        Oversampler can be either performance only or performance and language

        Args:
            sequences (list): sequences of interaction
            labels (list): target
            oversampler (list): list of the attributes by which to oversample, corresponding to the entries in x
            sampling_strategy (dict): dictionary with the keys as classes, and the values as number of samples to get

        Returns:
            oversampled_sequences: all the data you want to train with (original data + synthetic data or only synthetic data depending on the sampling strategy)
            oversampled_labels: associated performance labels to the oversampled_sequences
            oversampled_indices: indices of the final sequences (just to keep track what data comes from where.)
        """
        
        #Transformation of the data from vector of with decimals to vectors of only 0s and 1s
        new_sequences = []
        for i in range (len(sequences)):
            clicks=process_vectors(sequences[i])
            new_sequences.append(clicks)
        
        ###############################################################################################
        #1. check if we oversample on performance or performance and language based on the sampling strategy:
        if self._settings['ml']['oversampler']['number_categories'] == '2cat' : 
            #create 2 lists of sequences: one for 1 and one for 0 performance
            label_0=[]
            label_1=[]
            label_0_indices=[]
            label_1_indices=[]        
            for i, new_sequence in enumerate (new_sequences):
                if labels[i]==0:
                    label_0.append(new_sequence)
                    label_0_indices.append(i)
                if labels[i]==1:
                    label_1.append(new_sequence)
                    label_1_indices.append(i)   

                
        if self._settings['ml']['oversampler']['number_categories'] == '4cat':
            #create 4 lists of sequences: 0-french, 0-german, 1-french, 1-german
            label_0_french=[]
            label_0_german=[]
            label_1_french=[]
            label_1_german=[]
            label_0_french_indices=[]
            label_0_german_indices=[]
            label_1_french_indices=[]
            label_1_german_indices=[]
            for i, new_sequence in enumerate (new_sequences):
                if '0' in oversampler[i] and "Deutsch" in oversampler[i]:
                    label_0_german.append(new_sequence)
                    label_0_german_indices.append(i)
                if '0' in oversampler[i] and "Français" in oversampler[i]:
                    label_0_french.append(new_sequence)
                    label_0_french_indices.append(i)
                if '1' in oversampler[i] and "Deutsch" in oversampler[i]:
                    label_1_german.append(new_sequence)
                    label_1_german_indices.append(i)
                if '1' in oversampler[i] and "Français" in oversampler[i]:
                    label_1_french.append(new_sequence)
                    label_1_french_indices.append(i)

        ###############################################################################################
        oversampled_sequences = [] 
        oversampled_labels = [] 

        #2. get generation info: how many instances of which class to synthesize ?
        #3. generate the new instances
        if self._settings['ml']['oversampler']['rebalancing_mode'] == 'rebalance':
            oversampled_sequences+=sequences
            oversampled_labels+=labels  
            if self._settings['ml']['oversampler']['number_categories'] == '2cat':
                nb_fail, nb_success = get_generation_info('rebalance2cat', len_cat1= len(label_0), len_cat2 = len(label_1))
                generated_sequence, generated_label = complete_generation(nb0 =nb_fail, nb2 = nb_success, lst0=label_0, lst2=label_1)
                oversampled_sequences+= generated_sequence
                oversampled_labels+= generated_label
                
            if self._settings['ml']['oversampler']['number_categories'] == '4cat':
                nb0_french, nb0_german, nb1_french, nb1_german = get_generation_info('rebalance4cat', len_cat1= len(label_0_french), len_cat2 = len(label_0_german), len_cat3 = len(label_1_french), len_cat4=len(label_1_german))
                generated_sequence, generated_label = complete_generation(nb0 = nb0_french, nb1 = nb0_german, nb2 = nb1_french, nb3 = nb1_german, lst0=label_0_french, lst1=label_0_german, lst2=label_1_french, lst3=label_1_german)
                oversampled_sequences+= generated_sequence
                oversampled_labels+= generated_label

        if self._settings['ml']['oversampler']['rebalancing_mode'] == "100balanced":
            if self._settings['ml']['oversampler']['number_categories'] == '2cat':
                nb_fail, nb_success = get_generation_info('100balanced2cat', total_sample_size = self._settings['ml']['oversampler']['sample_size'])
                generated_sequence, generated_label = complete_generation(nb0 =nb_fail, nb2 = nb_success, lst0=label_0, lst2=label_1)
                oversampled_sequences+= generated_sequence
                oversampled_labels+= generated_label


            if self._settings['ml']['oversampler']['number_categories'] == '4cat':
                nb0_french, nb0_german, nb1_french, nb1_german= get_generation_info('100balanced4cat', total_sample_size = self._settings['ml']['oversampler']['sample_size'])
                generated_sequence, generated_label = complete_generation(nb0 = nb0_french, nb1 = nb0_german, nb2 = nb1_french, nb3 = nb1_german, lst0=label_0_french, lst1=label_0_german, lst2=label_1_french, lst3=label_1_german)
                oversampled_sequences+= generated_sequence
                oversampled_labels+= generated_label


        if self._settings['ml']['oversampler']['rebalancing_mode'] == "100unbalanced":
            if self._settings['ml']['oversampler']['number_categories'] == '2cat':
                nb_fail, nb_success = get_generation_info('100unbalanced2cat', len_cat1= len(label_0), len_cat2= len(label_1), total_sample_size = self._settings['ml']['oversampler']['sample_size'])
                generated_sequence, generated_label = complete_generation(nb0 =nb_fail, nb2 = nb_success, lst0=label_0, lst2=label_1)
                oversampled_sequences+= generated_sequence
                oversampled_labels+= generated_label
                
            if self._settings['ml']['oversampler']['number_categories'] == '4cat':             
                nb0_french, nb0_german, nb1_french, nb1_german= get_generation_info('100unbalanced4cat', len_cat1= len(label_0_french), len_cat2 = len(label_0_german), len_cat3 = len(label_1_french), len_cat4=len(label_1_german), total_sample_size=self._settings['ml']['oversampler']['sample_size'])
                generated_sequence, generated_label = complete_generation(nb0 = nb0_french, nb1 = nb0_german, nb2 = nb1_french, nb3 = nb1_german, lst0=label_0_french, lst1=label_0_german, lst2=label_1_french, lst3=label_1_german)
                oversampled_sequences+= generated_sequence
                oversampled_labels+= generated_label


        if self._settings['ml']['oversampler']['rebalancing_mode'] == '5050balanced':
            oversampled_sequences+=sequences
            oversampled_labels+=labels
            if self._settings['ml']['oversampler']['number_categories'] == '2cat':
                nb_fail, nb_success = get_generation_info('5050balanced2cat', len_cat1= len(label_0), len_cat2= len(label_1))
                generated_sequence, generated_label = complete_generation(nb0 =nb_fail, nb2 = nb_success, lst0=label_0, lst2=label_1)
                oversampled_sequences+= generated_sequence
                oversampled_labels+= generated_label


            if self._settings['ml']['oversampler']['number_categories'] == '4cat':
                nb0_french, nb0_german, nb1_french, nb1_german = get_generation_info('5050balanced4cat', len_cat1= len(label_0_french), len_cat2 = len(label_0_german), len_cat3 = len(label_1_french), len_cat4=len(label_1_german))
                generated_sequence, generated_label = complete_generation(nb0 = nb0_french, nb1 = nb0_german, nb2 = nb1_french, nb3 = nb1_german, lst0=label_0_french, lst1=label_0_german, lst2=label_1_french, lst3=label_1_german)
                oversampled_sequences+= generated_sequence
                oversampled_labels+= generated_label

        oversampled_indices  = list(range(0, len(oversampled_sequences)))
        return oversampled_sequences, oversampled_labels, oversampled_indices

    def sample(self, sequences:list, oversampler:list, labels:list, demographics:list) -> Tuple[list, list]:
        """Chooses the mode of oversampling
        
        PREVIOUS STRATEGIES:
        1. equal oversampling: All instances are oversampled by n, determined by imbalanced-learn
        2. Major oversampling: Only the largest class is oversampled
        3. Only Major Oversampling: Only the largest class is oversampled, all other classes are taken out the training set
        4. Minor oversampling: Only the smallest class is oversampled
        5. Only Minor Oversampling: Only the smallest class is oversampled, all other classes are taken out the training set

        ML4SCIENCE SAMPLING STRATEGIES:
        6. Rebalance: The minor category is oversampled to have the same size than the other
        7. 100unbalanced: All the data are synthetic, the proportions are the same as the initial data
        8. 100balanced: All the data are synthetic, equal number of instances in each category
        9. 5050balanced: 50% of the data are synthetic, equal number of instances in each category

        Args:
            sequences (list): behavioural sequences (features)
            oversampler (dict): column to use to oversample
            labels: labels linked to the machine learning part

        Returns:
            Tuple[list, list]: _description_
        """
        if self._settings['ml']['oversampler']['rebalancing_mode'] == 'equal_balancing':
            return self._equal_oversampling(sequences, oversampler, labels)

        elif 'major' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._major_oversampling(sequences, oversampler, labels)
        
        elif 'minor' in self._settings['ml']['oversampler']['rebalancing_mode']:
            return self._minor_oversampling(sequences, oversampler, labels)

        if self._settings['ml']['oversampler']['rebalancing_mode'] == 'rebalance':
            return self._oversample(sequences, labels, oversampler, 'rebalance')

        if self._settings['ml']['oversampler']['rebalancing_mode'] == '100unbalanced':
            return self._oversample(sequences, labels, oversampler, '100unbalanced')

        if self._settings['ml']['oversampler']['rebalancing_mode'] == '100balanced':
            return self._oversample(sequences, labels, oversampler, '100balanced')

        if self._settings['ml']['oversampler']['rebalancing_mode'] == '5050balanced':
            return self._oversample(sequences, labels, oversampler, '5050balanced')

    def get_indices(self) -> np.array:
        return self._indices
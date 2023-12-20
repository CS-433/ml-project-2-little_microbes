from keras import backend as K
from keras import losses
from keras.layers import Input, LSTM, Dense, Lambda, TimeDistributed
from keras.models import Model
import pickle
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import matplotlib.pyplot as plt
disable_eager_execution()
import random
from keras.models import load_model


###############################################################################################
# 1. DATA PROCESSING

def process_vectors(vectors):
    """
    Transforms click vectors in vectors of 0 and 1 (to ignore time)

    # Input
        vectors: array of size (N, d, 10) (sequences of d clicks for the N students)

    # Output
        result: array of size (N, d, 10) containing the processed vectors
    """
    result = []
    for vector in vectors:
        sorted_vector = sorted(enumerate(vector), key=lambda x: x[1], reverse=True)
        processed_vector = [0] * len(vector)
        processed_vector[sorted_vector[0][0]] = 1
        processed_vector[sorted_vector[1][0]] = 1
        result.append(processed_vector)
    
    return result


def get_text_data():
    """
    Loads and processes the data to train the model.

    # Input
        None

    # Output
        students : array of size (N, d, 10), containing the clicks for each student
        max_nb_clicks : int, largest number of clicks for one student
        num_encoder_clicks : int, number of different click possibilities+1 (+1 fake click for the decoder)
        inverse_clicks_dict : dict, click(string):number(int)
        input_clicks : dict, number(int):click(list)
        input_data : array of size (N, max_nb_clicks+1, num_encoder_clicks), containing the one hot vectors for each student
        decoder_input_data : array of size (N, max_nb_clicks+1, num_encoder_clicks), same as input_data but clicks are shifted by 1

    """
    students=[]
    false_first_click=[0]*10
    types_of_clicks=[false_first_click]
    with open('./data/ml4science_data_fake.pkl', 'rb') as file:
        data = pickle.load(file)

    for i in range (len(data['sequences'])):
        clicks=process_vectors(data['sequences'][i]['sequence'])
        students.append(clicks)

        for i, click in enumerate(clicks):
            if click not in types_of_clicks:
                types_of_clicks.append(click)

    input_clicks = sorted(types_of_clicks)
    num_encoder_clicks = len(input_clicks) 
    max_nb_clicks = max([len(clicks) for clicks in students]) + 1 

    print("Number of samples:", len(students))
    print("Number of unique input click types:", num_encoder_clicks-1) 
    print("Max number of clicks for one student:", max_nb_clicks-1)

    click_dict = {i: click for i, click in enumerate(input_clicks)} 

    clicks_as_strings=[]
    for click in input_clicks:
        string_list = [str(element) for element in click]

        result_string = ''.join(string_list)
        clicks_as_strings.append(result_string)
    
    inverse_click_dict = {click: i for i, click in enumerate(clicks_as_strings)} 
    encoder_input_data = np.zeros((len(students), max_nb_clicks, num_encoder_clicks), dtype="float32") 
    decoder_input_data = np.zeros((len(students), max_nb_clicks, num_encoder_clicks), dtype="float32")  
    
    for i, student in enumerate(students):
        decoder_input_data[i, 0, inverse_click_dict["0000000000"]] = 1.0

        for t, click in enumerate(student):
            string_list = [str(element) for element in click]
            string_click = ''.join(string_list)
            encoder_input_data[i, t, inverse_click_dict[string_click]] = 1.0
            decoder_input_data[i, t + 1, inverse_click_dict[string_click]] = 1.0

    return students, max_nb_clicks, num_encoder_clicks, input_clicks, inverse_click_dict, click_dict, encoder_input_data, decoder_input_data


def reformate(input, output):
    """
    Transforms the reconstruced output of the model to have only 0 and 1

    # Input
        input: array of shape (N, d, 23)
        output: array of shape (N, d, 23)

    # Output
        new_data: array of shape (N, d, 23), of only 1 and 0

    """
    new_data = np.zeros((input.shape))
    for i, student in enumerate(input):
        index_to_change = np.nonzero(input[i])[0]

        for j, number in enumerate(student):
            
            if j in index_to_change:
                new_data[i][j] = one_and_zeros_(output[i][j])
                
    return new_data

def one_and_zeros_(vector):
    """
    Replace each vector of the data array with a binary vector of 1 and 0. The largest value of the vector is set to 1 and the others to 0.
    
    # Input
        data: array, vector of length 23

    # Output
        transformed_data: array, vector of length 23 on only 1 and 0
    """
    transformed_data = np.zeros_like(vector)
    max_index = np.argmax(vector)
    transformed_data[max_index] = 1 
    
    return transformed_data


def back_to_ten(reconstruction, click_dict):
    """
    Replace every one-hot vector of size 23 by the corresponding vector of 10 values (as the initial data)
    
    # Input
        reconstruction: array of the one-hot vectors of size 23
        click_dict: dict, associates each type of vector to a number

    # Output
        reconstruction_final: array of the vectors of size 10 
    """
    reconstruction_final = np.zeros((reconstruction.shape[0], reconstruction.shape[1], 10))
    for i, student in enumerate(reconstruction):
        for j, vector in enumerate(student):
            index_max = np.argmax(reconstruction[i][j])
            reconstruction_final[i][j] = click_dict[index_max]
    return reconstruction_final


#Click_dict is a dictionnary that associates any kind of vector to a number
#Generated by the function get_text_data 
click_dict = {0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 2: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 3: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0], 4: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 5: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 6: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 7: [0, 0, 1, 0, 0, 0, 0, 0, 0, 1], 8: [0, 0, 1, 0, 0, 0, 0, 0, 1, 0], 9: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 10: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 11: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 12: [0, 1, 0, 0, 0, 0, 0, 0, 0, 1], 13: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], 14: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0], 15: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0], 16: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0], 17: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], 18: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 19: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0], 20: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0], 21: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0], 22: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]}


########################################################################################
# 2. MODEL CREATION AND TESTING

def create_lstm_vae(input_dim, batch_size, intermediate_dim, latent_dim):
    """
    Creates an LSTM variational encoder.

    # Input
        input_dim: int
        batch_size: int
        intermediate_dim: int
        latent_dim: int

    # Output
        vae: Model
        encoder: Model
        generator: Model
        stepper: Model
        
    """
    x = Input(shape=(None, input_dim,)) 
    # LSTM encoding
    h = LSTM(units=intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(units=latent_dim)(h)
    z_log_sigma = Dense(units=latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    z_reweighting = Dense(units=intermediate_dim, activation="linear")
    z_reweighted = z_reweighting(z)

    # "next-click" data for prediction
    decoder_words_input = Input(shape=(None, input_dim,)) 

    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True, return_state=True)

    h_decoded, _, _ = decoder_h(decoder_words_input, initial_state=[z_reweighted, z_reweighted])
    decoder_dense = TimeDistributed(Dense(input_dim, activation="softmax")) 
    decoded_onehot = decoder_dense(h_decoded)

    # end-to-end autoencoder
    vae = Model([x, decoder_words_input], decoded_onehot)

    # encoder, from inputs to latent space
    encoder = Model(x, [z_mean, z_log_sigma])

    # generator, from latent space to reconstructed inputs -- for inference's first step
    decoder_state_input = Input(shape=(latent_dim,))
    _z_rewighted = z_reweighting(decoder_state_input)
    _h_decoded, _decoded_h, _decoded_c = decoder_h(decoder_words_input, initial_state=[_z_rewighted, _z_rewighted])
    _decoded_onehot = decoder_dense(_h_decoded)
    generator = Model([decoder_words_input, decoder_state_input], [_decoded_onehot, _decoded_h, _decoded_c])

    # RNN for inference
    input_h = Input(shape=(intermediate_dim,))
    input_c = Input(shape=(intermediate_dim,))
    __h_decoded, __decoded_h, __decoded_c = decoder_h(decoder_words_input, initial_state=[input_h, input_c])
    __decoded_onehot = decoder_dense(__h_decoded)
    stepper = Model([decoder_words_input, input_h, input_c], [__decoded_onehot, __decoded_h, __decoded_c])

    def vae_loss(x, x_decoded_onehot):
        xent_loss = losses.categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss
    
    vae.compile(optimizer="adam", loss=vae_loss)
    vae.summary()

    return vae, encoder, generator, stepper



def new_accuracy(input, output):
    """
    Computes the accuracy by comparing each click of the output with the expected click in the input.
    
    # Input
        input: array of shape (N, d, 23) (initial data)
        output: array of shape (N, d, 24) (reconstructed data)

    # Output
        accuracy: int, % of accuracy of the model
    """
    trues=0
    falses=0
    nb_clicks=[]
    for i in range(len(input)):
        for j in range (len(input[i])):
            if np.all(input[i, j, :] == 0):
                nb_clicks.append(j)
                break
    for i, student in enumerate(input):
        for j in range(nb_clicks[i]):
            if np.all(input[i][j]==output[i][j]):
                trues+=1
            else:
                falses+=1
    accuracy=100*trues/(trues+falses)
    print("The accuracy of the model is", accuracy, "%")
    return accuracy


########################################################################################
# 3. SYNTHETIC DATA GENERATION

def get_generation_info(type_of_split, len_cat1=None, len_cat2=None, len_cat3 = None, len_cat4= None, total_sample_size = None):
    '''
    Gives the generation information needed depending on the type of oversampling
    
    # Input
        type_of_split: str, sampling strategy
        len_cat(i): int, number of instances in the category (i)
        total_sample_size: int, total number of instances we want in the final dataset (only for 100% synthetic data strategies)         

    # Output
        nb(i): int, number of instances to generate from the category (i)   
    '''

    #rebalance for 2 cat
    if type_of_split == 'rebalance2cat' and len_cat1!=None and len_cat2!=None and len_cat3==None and len_cat4==None: 
        diff = len_cat1 - len_cat2
        if diff > 0:
            print('Oversample category 2 by', diff)
            return 0, diff
        if diff < 0:
            print('Oversample category 1 by', -diff)
            return -diff,0
        if diff == 0:
            print('Already balanced')
            return [0,0]
    #rebalance for 4 cat
    if type_of_split == "rebalance4cat" and len_cat1!=None and len_cat2!=None and len_cat3!=None and len_cat4!=None:
        lst_cat=[len_cat1, len_cat2, len_cat3, len_cat4]
        max_len=np.max(lst_cat)
        nbs=[]
        for i, length in enumerate(lst_cat):
            if length<max_len:
                print("Generate", max_len-length, f"new instances of cat{i+1}")
                nbs.append(max_len-length)
            else:
                nbs.append(0)
        return (nbs)
                
        
    #100imbalanced for 2 cat
    if type_of_split == '100unbalanced2cat' and total_sample_size != None and len_cat3==None and len_cat4==None and len_cat1!=None and len_cat2!=None:
        perc1 = np.round(len_cat1 *100 / (len_cat1 + len_cat2), 0)
        perc2 = 100 - perc1
        nb1 = np.round(perc1/100 * total_sample_size)
        nb2 = total_sample_size - nb1
        print('Generate ', int(nb1), 'new instances of cat1 and ', int(nb2), 'new instances of cat2')
        return [int(nb1), int(nb2)]
    #100imbalanced for 4 cat
    if type_of_split == '100unbalanced4cat' and total_sample_size != None and len_cat3!=None and len_cat4!=None and len_cat1!=None and len_cat2!=None:
        perc1 = int(np.round(len_cat1 *100 / (len_cat1 + len_cat2 + len_cat3 + len_cat4), 0))
        perc2 = int(np.round(len_cat2 *100 / (len_cat1 + len_cat2 + len_cat3 + len_cat4), 0))
        perc3 = int(np.round(len_cat3 *100 / (len_cat1 + len_cat2 + len_cat3 + len_cat4), 0))
        perc4= int(100-(perc1+perc2+perc3))
        nb1 = int(np.round(perc1/100 * total_sample_size))
        nb2 = int(np.round(perc2/100 * total_sample_size))
        nb3 = int(np.round(perc3/100 * total_sample_size))
        nb4 = int(total_sample_size - (nb1+nb2+nb3))
        print('Generate ', nb1, ' of cat1,', nb2, ' of cat2,', nb3, "of cat3 and", nb4, "of cat4")
        return[nb1, nb2, nb3, nb4]
        
    #100 balanced for 2 cat
    if type_of_split == '100balanced2cat' and total_sample_size != None :
        print("Generate", int(np.round(total_sample_size/2, 0)), "new instances of categories 1 and 2")
        return [int(np.round(total_sample_size/2, 0)), int(np.round(total_sample_size/2, 0))]
    #100 balanced for 4 cat
    if type_of_split == '100balanced4cat' and total_sample_size != None :
        one_quarter= int(np.round(total_sample_size/4, 0))
        print("Generate", one_quarter, "new instances of categories 1, 2, 3 and 4")
        return [one_quarter, one_quarter, one_quarter, one_quarter]

    #5050 balanced for 2 cat    
    if type_of_split == '5050balanced2cat':
        print('Generate', len_cat2, ' new instances of category 1, and ', len_cat1, ' new instances of category 2')
        return[len_cat2, len_cat1]

    #5050 balanced for 4 cat
    if type_of_split == '5050balanced4cat':
        total_sample_size = (len_cat1 + len_cat2 + len_cat3 + len_cat4) * 2
        size_each = np.round(total_sample_size / 4, 0)
        
        if len_cat1 < size_each:
            nb1 = size_each - len_cat1
            print('Generate ', nb1, ' new instances of category 1')
        else: 
            nb1 = size_each
            print('Take only ', size_each, ' instances of category 1')
            
        if len_cat2 < size_each:
            nb2 = size_each - len_cat2
            print('Generate ', nb2, ' new instances of category 2')
        else: 
            nb2 = size_each
            print('Take only ', nb2, ' instances of category 2')
            
        if len_cat3 < size_each:
            nb3 = size_each - len_cat3
            print('Generate ', nb3, ' new instances of category 3')
        else: 
            nb3 = size_each
            print('Take only ', nb3, ' instances of category 3')
            
        if len_cat4 < size_each:
            nb4 = size_each - len_cat4
            print('Generate ', nb4, ' new instances of category 4')
        else: 
            nb4 = size_each
            print('Take only ', nb4, ' instances of category 4')
        return [int(nb1), int(nb2), int(nb3), int(nb4)]


def generation(input_to_copy, click_dict, nb_random_indices = None):
    """
    Generation of synthetic data from initial data with pre-trained encoder and generator (in .h5 files)
    
    # Input
        input_to_copy: list of lists, the initial sequences we want to use to generate new similar instances
        click_dict: dictionnary that associates each kind of sequence to a number 
        nb_random_indices: int, number of new instances we want to generate         

    # Output
        final_list: list of lists, the new instances that were generated
    """
    #Parameters of the model
    input_dim=23
    intermediate_dim = 64
    latent_dim = 2
    batch_size = 1

    #Creation of the encoder and generator based on the weights of the saved .h5 files
    vae, encoder, generator, stepper = create_lstm_vae(input_dim, batch_size, intermediate_dim, latent_dim)
    encoder.load_weights('ml/samplers/weights_files/encoder_weights_final.h5')
    generator.load_weights('ml/samplers/weights_files/generator_weights_final.h5')
    
    #if more sample to create than original data
    if nb_random_indices != None and int(nb_random_indices) > len(input_to_copy):

        #Transforms 10-vector -> 23-one-hot-vector
        max_length = max(len(element) for element in input_to_copy)
        input = np.zeros((len(input_to_copy), max_length, 23))
        for i, list in enumerate(input_to_copy):
            for j, vector in enumerate(list):
                for key, value in click_dict.items():
                    if value == vector:
                        input[i][j][key] = 1
        
        #Transforms list of list -> array
        seed_sequence = np.zeros((len(input), max_length, 23))
        for i, list in enumerate(input):
            seed_sequence[i, :len(list)] = list
        
        #First create the same number of instances as the input
        max_nb_clicks = max([len(clicks) for clicks in seed_sequence])
        sequence_length = max_nb_clicks
        z_mean, z_log_sigma = encoder.predict(seed_sequence)
        epsilon = np.random.normal(size = z_mean.shape)
        latent_state = z_mean + np.exp(0.5 * z_log_sigma) * epsilon 
        generated_sequence, updated_hidden_state, updated_cell_state = generator.predict([seed_sequence, latent_state])


        #Then add more instances
        new_seed_sequences = seed_sequence
        for i in range(int(nb_random_indices)- len(input_to_copy) ):
            index_1 = random.randint(0, len(z_mean) - 1)  
            index_2 = random.randint(0, len(z_mean) - 1)
            new_z_mean = np.array([(x + y) / 2 for x, y in zip(z_mean[index_1], z_mean[index_2])])
            new_z_sigma = np.array([(x + y) / 2 for x, y in zip(z_log_sigma[index_1], z_log_sigma[index_2])])
            index_3 = random.randint(0, len(seed_sequence) - 1)
            new_seed_sequence = np.expand_dims(seed_sequence[index_3], axis = 0)
            new_epsilon = np.random.normal(size = new_z_mean.shape)
            new_latent_state = new_z_mean + np.exp(0.5 * new_z_sigma) * new_epsilon
            new_latent_state = np.expand_dims(new_latent_state, axis = 0)
            new_generated_sequence, _, _ = generator.predict([new_seed_sequence, new_latent_state])
            generated_sequence = np.vstack([generated_sequence, new_generated_sequence])
            new_seed_sequences = np.vstack([new_seed_sequences, new_seed_sequence])

        reformated_output = reformate(new_seed_sequences, generated_sequence)
        reformated_output = back_to_ten(reformated_output, click_dict)
        #Transforms array -> list of list
        final_list = []
        for i in range(reformated_output.shape[0]):
            inter_list = []
            for j in range(reformated_output.shape[1]):
                if any(element != 0 for element in reformated_output[i][j]):
                    inter_list.append(reformated_output[i][j].tolist())
            final_list.append(inter_list)
    
        return final_list   

    #if less to generate than original data
    if nb_random_indices != None:
        random_indices = np.random.choice(len(input_to_copy), size = int(nb_random_indices), replace = False)
        input_to_copy = [input_to_copy[indice] for indice in random_indices]

        #Transforms 10-vector -> 23-one-hot-vector
        max_length = max(len(element) for element in input_to_copy)
        input = np.zeros((len(input_to_copy), max_length, 23))
        for i, list in enumerate(input_to_copy):
            for j, vector in enumerate(list):
                for key, value in click_dict.items():
                    if value == vector:
                        input[i][j][key] = 1
        
        #Transforms list of list -> array
        seed_sequence = np.zeros((len(input), max_length, 23))
        for i, list in enumerate(input):
            seed_sequence[i, :len(list)] = list

    #if same number to generate than initial data
    else:
        #Transforms 10-vector -> 23-one-hot-vector
        max_length = max(len(element) for element in input_to_copy)
        input = np.zeros((len(input_to_copy), max_length, 23))
        for i, list in enumerate(input_to_copy):
            for j, vector in enumerate(list):
                for key, value in click_dict.items():
                    if value == vector:
                        input[i][j][key] = 1
        
        #Transforms list of list -> array
        seed_sequence = np.zeros((len(input), max_length, 23))
        for i, list in enumerate(input):
            seed_sequence[i, :len(list)] = list

    max_nb_clicks = max([len(clicks) for clicks in seed_sequence])
        
    sequence_length = max_nb_clicks
    z_mean, z_log_sigma = encoder.predict(seed_sequence)
    epsilon = np.random.normal(size = z_mean.shape)
    latent_state = z_mean + np.exp(0.5 * z_log_sigma) * epsilon # 64 values for each student
    generated_sequence, updated_hidden_state, updated_cell_state = generator.predict([seed_sequence, latent_state])

    reformated_output = reformate(seed_sequence, generated_sequence)
    reformated_output = back_to_ten(reformated_output, click_dict)

    #Transforms array -> list of list
    final_list = []
    for i in range(reformated_output.shape[0]):
        inter_list = []
        for j in range(reformated_output.shape[1]):
            if any(element != 0 for element in reformated_output[i][j]):
                inter_list.append(reformated_output[i][j].tolist())
        final_list.append(inter_list)

    return final_list   



def complete_generation(nb0 = 0, nb1 = 0, nb2 = 0, nb3 = 0, lst0=None, lst1=None, lst2=None, lst3=None):
    """
    Intermediate generation process that takes the information from get_generation_info and calls generation in order to create instances of each categories
    
    # Input
        nb(i): int, number of instances to generate from the category (i)
        lst(i): list, initial instances of the category (i)      

    # Output
        generated_sequence: list, Generated instances 
        generated_label: list, Performance labels of the generated instances
    """
    #if only oversample on labels : order of the categories = 0-None-1-None
    #if oversample on labels and languages : order of the categories = French0 - German0 -French1 - German1
    nbs = [nb0, nb1, nb2, nb3]
    lsts = [lst0, lst1, lst2, lst3]
    labels = [0,0,1,1]
    generated_sequence = []
    generated_label = []
    
    for i in range(len(nbs)):
        if nbs[i] != 0:
            generated_sequence += generation(lsts[i], click_dict, nbs[i])
            generated_label += [labels[i]]*nbs[i]
            
    return generated_sequence, generated_label
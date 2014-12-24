# -*- coding: utf-8 -*-
import LoadData


dataset_matrix, label_vector, dataset_matrix_r, label_vector_r = LoadData.preprocess_data()

datasets = LoadData.load_data_multi(dataset_matrix_r, label_vector_r)



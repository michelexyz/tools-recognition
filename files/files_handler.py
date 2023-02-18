
import os
import pathlib
#os.path.join('app', 'subdir', 'dir', 'filename.foo')


#TODO ma queste righe vengono rieseguite ogni volta che il file viene importato?
parent_folder = pathlib.Path(__file__).parent

project_folder = parent_folder.parent

"""DATASET PATH"""
dataset_name = 'data'



dataset_path = project_folder.joinpath(dataset_name)

processed_name = 'processed'

processed_path = dataset_path.joinpath(processed_name)

#Project path the only thing that needs to be changed if clone the project
#project_path = '/Users/michelevannucci/PycharmProjects/ToolsRecognition'

#Working files path
#
#The files that will be genarated by the different method to be used by the others of the pipeline
#w_files_path = '/files/working_files/'



"""WORKING FOLDER PATH"""
w_folder_name = 'working_files'

#Working files absolute path
w_files_path = parent_folder.joinpath(w_folder_name)

def get_file_abs_path(file_name):
    return w_files_path.joinpath(file_name).resolve()

def get_abs_path(folder_name, file_name):
    return parent_folder.joinpath(folder_name,file_name).resolve()

def get_processed_path():
    return processed_path.resolve()

def get_data_folder(folder):
    folder_path = dataset_path.joinpath(folder)
    return folder_path



if __name__ == '__main__':
    path = get_abs_path('graphs', 'tree')
    print(path)
